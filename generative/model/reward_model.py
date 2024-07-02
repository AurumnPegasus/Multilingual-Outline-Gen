import pytorch_lightning as pl
from transformers import MBartForConditionalGeneration, MT5ForConditionalGeneration, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer, BloomModel, AutoTokenizer
import torch
from icecream import ic
from rouge import Rouge
import json
from indicnlp.transliterate import unicode_transliterate
import pandas as pd
import re

def fsaReward(predictions, fsa_dict, lang, dom):
    intro_map = {
        'bn': 'ভূমিকা',
        'en': 'Introduction',
        'hi': 'परिचय',
        'kn': 'ಪರಿಚಯ',
        'ml': 'ആമുഖം',
        'mr': 'परिचय',
        'or': 'ପରିଚୟ',
        'pa': 'ਜਾਣ-ਪਛਾਣ',
        'ta': 'அறிமுகம்',
        'te': 'పరిచయం'
    }

    freward = []
    for i in range(len(predictions)):
        preds = predictions[i]
        ln = lang[i]
        dm = dom[i]

        preds = preds.replace(ln, '')

        ln = ln[:2]
        preds  = re.sub(r'<extra_id_\d*>', '', preds)
        preds = f'<source> {preds} <sink>'
        # preds = preds.replace('<SEP>','*')
        titles = preds.split(' ')

        probs = 1
        steps = 0
        first = titles[0]
        for i in range(1, len(titles)):
            if titles[i] == '':
                continue

            probs = probs*(fsa_dict[ln][dm][first][titles[i]])
            first = titles[i]
            steps += 1

        freward.append(1 - probs**(1/steps))
        # freward.append(probs**(1/steps))

    # ic(freward, predictions)
    return freward


class Summarizer(pl.LightningModule):
    def __init__(self, fsa_dict, reward_val, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.rouge = Rouge()
        # ic(self.hparams.keys())
        self.fsa_dict = fsa_dict
        self.reward_val = reward_val
        self.config = AutoConfig.from_pretrained(self.hparams.config)

        if self.hparams.is_mt5 == 1:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

        self.languages_map = {
            'bn': 'bn_IN',
            'de': 'de_DE',
            'en': 'en_XX',
            'es': 'es_XX',
            'fr': 'fr_XX',
            'gu': 'gu_IN',
            'hi': 'hi_IN',
            'it': 'it_IT',
            'kn': 'kn_IN',
            'ml': 'ml_IN',
            'mr': 'mr_IN',
            'or': 'or_IN',
            'pa': 'pa_IN',
            'ta': 'ta_IN',
            'te': 'te_IN',
        }

        # self.optimizer = self.configure_optimizers()


    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def _step(self, batch):
        input_ids, attention_mask, labels, lang, domain = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['lang'], batch['domain']
        outputs = self(input_ids.clone(), attention_mask.clone(), labels.clone())
        loss = outputs['loss']
        logits = outputs['logits']
        inp_text, pred_text, ref_text, lang, dom = self._generative_step(batch)
        reward = fsaReward(pred_text, self.fsa_dict, lang, dom)
        reward = torch.FloatTensor(reward).to(self.device)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        sample_idx = torch.zeros(probs.size(0), probs.size(1))
        sample_probs = torch.zeros(probs.size(0), probs.size(1))
        batch_list = []
        # ic(probs.shape)

        for i, s in enumerate(probs):
            temp_idx = torch.multinomial(s, 1)
            temp_probs = s.gather(1, temp_idx)
            # sample_probs = temp_probs.squeeze(1)
            # ic(s.shape, sample_probs.shape)
            # ic(temp_probs.shape)
            batch_list.append(temp_probs)

        batch_list = torch.cat(batch_list, dim=1)
        # ic(batch_list.shape)
        batch_list = torch.transpose(batch_list, 0, 1)
        # ic(batch_list.shape)
        # sample_probs = sample_probs.contiguous()
        sample_logprobs = torch.log(batch_list)


        # argmax = torch.amax(probs, dim=2)
        # batch_size, max_len = argmax.size()
        # bestaction = torch.log
        fsa_reward = reward.unsqueeze(1)

        # ic(sample_probs.shape)
        # ic(sample_logprobs.shape)

        # fsa_loss = -sample_logprobs * fsa_reward
        # fsa_loss = fsa_loss.sum(-1).mean()

        sample_loss = -sample_logprobs * fsa_reward
        sample_loss = sample_loss.sum(-1).mean()
        # ic(sample_loss, loss, fsa_reward)

        # ic(loss, fsa_loss, fsa_reward, self.reward_val)

        loss = loss + self.reward_val*sample_loss
        # ic(self.reward_val, loss)
        # ic(ref_text, pred_text, loss)
        return loss

    def _generative_step(self, batch):

        if not self.hparams.is_mt5:
            try:
                token_id = self.hparams.tokenizer.lang_code_to_id[batch['lang']]
                self.hparams.tokenizer.tgt_lang = batch['lang']
            except:
                token_id = 250044
                self.hparams.tokenizer.tgt_lang = 'ta_IN'

            generated_ids = self.model.generate(
                             input_ids=batch['input_ids'],
                             attention_mask=batch['attention_mask'],
                             use_cache=True,
                             num_beams=self.hparams.eval_beams,
                             forced_bos_token_id=token_id,
                             max_length=self.hparams.tgt_max_seq_len #understand above 3 arguments
             )
        else:
            self.hparams.tokenizer.tgt_lang = batch['lang']
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=True,
                num_beams=self.hparams.eval_beams,
                max_length=self.hparams.tgt_max_seq_len
            )


        input_text = self.hparams.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        pred_text = self.hparams.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if self.hparams.is_mt5 == 1:
            batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
        ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        return input_text, pred_text, ref_text, batch['lang'], batch['domain']


    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, on_epoch=True)
        return {'val_loss': loss}


    def validation_epoch_end(self, outputs):
        return

    def predict_step(self, batch, batch_idx):
        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def get_native_text_from_unified_script(self, unified_text, lang):
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(unified_text, "hi", lang)

    def process_for_rouge(self, text, lang):
        native_text = text
        if lang!='en':
            # convert unified script to native langauge text
            native_text = self.get_native_text_from_unified_script(text, lang)

        native_text = native_text.strip()
        # as input and predicted text are already space tokenized
        native_text = ' '.join([x for x in native_text.split()])
        return native_text

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'test_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def test_epoch_end(self, outputs):
        df_to_write = pd.DataFrame(columns=['lang', 'input_text', 'ref_text', 'pred_text', 'rouge'])
        input_text = []
        langs = []
        pred_text = []
        ref_text = []
        langs = []

        for x in outputs:
            input_texts.extend(x['input_text'])
            pred_texts.extend(x['pred_text'])
            ref_texts.extend(x['ref_text'])
            langs.extend(x['lang'])

        for key in self.languages_map:
            self.languages_map[key]['original_pred_text'] = [self.process_for_rouge(pred_text, self.lang_id_map[lang]) for pred_text, lang in zip(pred_texts, langs) if lang == self.languages_map[key]['id']]
            self.languages_map[key]['original_ref_text'] = [self.process_for_rouge(ref_text, self.lang_id_map[lang]) for ref_text, lang in zip(ref_texts, langs) if lang == self.languages_map[key]['id']]
            self.languages_map[key]['original_input_text'] = [self.process_for_rouge(input_text, self.lang_id_map[lang]) for input_text, lang in zip(input_texts, langs) if lang == self.languages_map[key]['id']]

        overall_rouge = 0
        for key in self.languages_map:
            try:
                self.languages_map[key]['rouge'] = self.rouge.get_scores(self.languages_map[key]['original_pred_text'], [self.languages_map[key]['original_ref_text']]).score
                self.log(f"test_rouge_{key}", self.languages_map[key]['rouge'])
                overall_rouge += self.languages_map[key]['rouge']
            except:
                pass

        self.log("test_rouge", overall_rouge/len(self.languages_map))

        for key in self.languages_map:
            l = len(self.languages_map[key]['original_pred_text'])
            self.languages_map[key]['rouges'] = [self.cal_bleu.corpus_score([self.languages_map[key]['original_pred_text'][i]], [[self.languages_map[key]['original_ref_text'][i]]]).score for i in range(len(self.languages_map[key]['original_pred_text']))]
            df_key = pd.DataFrame({
                'lang':[key for i in range(l)],
                'input_text':[self.languages_map[key]['original_input_text'][i] for i in range(l)],
                'pred_text':[self.languages_map[key]['original_pred_text'][i] for i in range(l)],
                'ref_text':[self.languages_map[key]['original_ref_text'][i] for i in range(l)],
                'rouge':[self.languages_map[key]['rouges'][i] for i in range(l)]
            })
            df_to_write = pd.concat([df_to_write, df_key])

        if self.hparams.is_mt5 == 1:
            df_to_write.to_csv(self.hparams.prediction_path + 'preds_mt5.csv', index=False)
        else:
            df_to_write.to_csv(self.hparams.prediction_path + 'preds_mbart.csv', index=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Bart Fine-tuning Parameters')
        parser.add_argument('--learning_rate', default=2e-5, type=float)
        parser.add_argument('--model_name_or_path', default='bart-base', type=str)
        parser.add_argument('--eval_beams', default=4, type=int)
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
        parser.add_argument('--tokenizer', default='bart-base', type=str)
        return parent_parser

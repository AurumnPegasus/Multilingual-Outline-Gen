import requests

def content_type(url):
    try:
        r = requests.get(url)
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        return 'na'
    content_type = r.headers.get('content-type')

    try:
        if 'application/pdf' in content_type:
            return 'pdf'

        elif 'text/html' in content_type:
            return 'html'

    except Exception as e:
        print(e)
        return 'na'

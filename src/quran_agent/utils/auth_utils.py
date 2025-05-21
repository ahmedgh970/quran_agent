import requests

def get_access_token(
    auth_url,
    client_id,
    client_secret
) -> str:
    """
    Get the access token for the Quran API.
    """
    response = requests.post(
        url=auth_url,
        auth=(client_id, client_secret),
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        data='grant_type=client_credentials&scope=content'
    )
    return response.json()['access_token']
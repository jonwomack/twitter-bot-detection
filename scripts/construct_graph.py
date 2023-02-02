import requests
import os
import json

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER_TOKEN")


def create_url_followers(user_id):
    # Replace with user ID below
    return "https://api.twitter.com/2/users/{}/followers".format(user_id)

def create_url_following(user_id):
    # Replace with user ID below
    return "https://api.twitter.com/2/users/{}/following".format(user_id)

def get_params():
    return {"user.fields": "created_at"}

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FollowersLookupPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def get_followers(user_id):
    # Followers Request
    url = create_url_followers(user_id)
    params = get_params()
    json_response = connect_to_endpoint(url, params)
    follower_ids = []
    for follower in json_response['data']:
        # print(follower['name'])
        # print(follower['id'])
        follower_ids.append(follower['id'])
    return follower_ids
    
def get_following(user_id):
    # Following Request
    url = create_url_following(user_id)
    params = get_params()
    json_response = connect_to_endpoint(url, params)
    following_ids = []
    for following in json_response['data']:
        # print(following['name'])
        # print(following['id'])
        following_ids.append(following['id'])
    return following_ids
    # print(type(json.dumps(json_response, indent=4, sort_keys=True)))

def main():
    initial_user = 1438984390256721926
    user_ids = [initial_user]
    explored_user_ids = []
    while len(user_ids) > 0:
        current_user_id = user_ids.pop(0)
        explored_user_ids.append(current_user_id)
        print(current_user_id)
        followers = get_followers(current_user_id)
        following = get_following(current_user_id)
        intersection = list(set(followers) & set(following))
        user_ids = user_ids + intersection
        print(intersection)
    
    print(explored_user_ids)





if __name__ == "__main__":
    main()

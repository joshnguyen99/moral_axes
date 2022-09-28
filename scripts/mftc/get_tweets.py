# Source: https://osf.io/mzg5w
import json
import sys
from twython import Twython
import time

# Fill your Twitter API key and access token below
# Information can be found on the Twitter developer website:
# https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html
consumer_key = ''
consumer_secret = ''
access_token_key = ''
access_token_secret = ''


def parse(input, output, corpus_name="all"):
    try:
        temp = open(input)
    except:
        print("Please enter a valid input file")

    data = json.load(temp)
    counter = 0

    for d in data:
        if((corpus_name != "all") and (corpus_name != d["Corpus"])):
            continue
        # Skip over corpus without tweet ID's
        if(d["Corpus"] == "Davidson"):
            if(corpus_name == "Davidson"):
                print("Tweet text is not availble for the Davidson corpus. "
                      "You can use the tweet_ids and retrieve the text from "
                      "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv")
            continue
        for t in d["Tweets"]:
            id = t["tweet_id"]
            # retrieves tweet text if available
            try:
                text, date = call_twitter_api(id)
                print(f"Retrieved tweet text for tweet ID: {str(id)} Tweet = {text[:10]}")
                t["tweet_text"] = text
                t["date"] = date
            except:
                print(f"NOT FOUND ID: {str(id)}")
                t["tweet_text"] = 'no tweet text available'
                t["date"] = 'no date available'

            # increment counter
            counter += 1

            # In order to deal with the Twitter API access limits
            if counter == 900:
                print("Twitter API has timed out. Waiting 15 minutes before resuming script")
                time.sleep(900)

            # reodering the json elements
            temp = t["annotations"]
            t.pop("annotations")
            t["annotations"] = temp

    check = False
    if(corpus_name != "all"):
        for d in data:
            if(d["Corpus"] == corpus_name):
                data = d
                check = True
                break
        if(check == False):
            print("Please enter a valid corpus name")

    with open(output, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def call_twitter_api(id):
    twitter = Twython(consumer_key, consumer_secret, access_token_key, access_token_secret)
    tweet = twitter.show_status(id=id)
    return tweet['text'], tweet['created_at']


if __name__ == "__main__":
    try:
        parse(sys.argv[1], sys.argv[2], sys.argv[3])
    except:
        try:
            parse(sys.argv[1], sys.argv[2])
        except Exception:
            print("Please use the following command to run: "
                  "python text_script.py <inputfile> <outputfile> <OPTIONAL corpusname>")
            exit(1)

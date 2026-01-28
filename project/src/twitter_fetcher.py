# twitter_fetcher.py
import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

class TwitterFetcher:
    def __init__(self):
        # Get Twitter API credentials from environment
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.consumer_key = os.getenv('TWITTER_API_KEY')
        self.consumer_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        
        # Initialize client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_secret,
            wait_on_rate_limit=True
        )
    
    def fetch_user(self, username):
        """Fetch user data from Twitter API"""
        try:
            # Remove @ if present
            username = username.lstrip('@')
            
            # Get user data
            user = self.client.get_user(
                username=username,
                user_fields=[
                    'public_metrics',
                    'description',
                    'verified',
                    'created_at'
                ]
            )
            
            if user.data:
                return {
                    'username': username,
                    'followers_count': user.data.public_metrics['followers_count'],
                    'following_count': user.data.public_metrics['following_count'],
                    'tweet_count': user.data.public_metrics['tweet_count'],
                    'verified': user.data.verified,
                    'description': user.data.description or '',
                    'created_at': user.data.created_at.isoformat() if user.data.created_at else '',
                    'name': user.data.name
                }
            else:
                return {'error': 'User not found'}
                
        except tweepy.errors.NotFound:
            return {'error': 'User not found'}
        except Exception as e:
            return {'error': str(e)}
    
    def fetch_multiple_users(self, usernames):
        """Fetch multiple users"""
        results = []
        for username in usernames:
            results.append(self.fetch_user(username))
        return results

# Example usage
if __name__ == "__main__":
    fetcher = TwitterFetcher()
    
    # Test with your example
    profile = {
        "user_id": 1574721129670910000,
        "followers_count": 3910,
        "following_count": 5288,
        "tweet_count": 2844,
        "account_age_days": 1160,
        "verified": False,
        "description": "Tech enthusiast",
        "created_at": "2021-10-01T12:00:00Z"
    }
    
    # This profile can now be sent to /predict endpoint
    print("Test profile ready for API")
"""Mock modules for testing."""

# Mock for dotenv
class MockLoadDotenv:
    def __call__(self, *args, **kwargs):
        pass

load_dotenv = MockLoadDotenv()

# Mock for OpenAI
class MockOpenAI:
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    class Chat:
        class Completions:
            @staticmethod
            def create(*args, **kwargs):
                return MagicMockResponse()
    
    chat = Chat()

class MagicMockResponse:
    def __init__(self):
        self.choices = [
            MagicMockChoice()
        ]

class MagicMockChoice:
    def __init__(self):
        self.message = MagicMockMessage()

class MagicMockMessage:
    def __init__(self):
        self.content = '{"name": "error-bars-defined", "panels": [{"panel_label": "A", "error_bar_on_figure": "yes", "error_bar_defined_in_legend": "yes", "error_bar_meaning": "standard deviation", "from_the_caption": "Error bars indicate mean±s.d."}]}'

# Create the OpenAI class
OpenAI = MockOpenAI
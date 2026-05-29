import json
import logging
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("similarity-api")

def preprocess_strings(string_a: str, string_b: str) -> dict:
    """
    Use AWS Bedrock Converse API to preprocess strings.
    Expands abbreviations, removes unwanted plurals (like 'banks' to 'bank'),
    and determines if they are similar.
    
    Returns a dict:
    {
        "is_similar": bool,
        "string_a_normalized": str,
        "string_b_normalized": str
    }
    """
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        system_prompts = [{
            "text": (
                "You are a string preprocessing assistant for a similarity matching pipeline. "
                "Your task is to analyze two strings and:\n"
                "1. Expand any abbreviations to what they mean (e.g., 'Md.' to 'Mohammed').\n"
                "2. Remove unwanted things like an 's' at the end of a bank name (e.g. 'banks' to 'bank').\n"
                "3. Determine if the two strings are similar or refer to the exact same entity.\n\n"
                "You must return ONLY a strict JSON object with this exact schema:\n"
                "{\n"
                "  \"is_similar\": true or false,\n"
                "  \"string_a_normalized\": \"<normalized version of first string>\",\n"
                "  \"string_b_normalized\": \"<normalized version of second string>\"\n"
                "}\n"
                "Do not include markdown blocks or any other text."
            )
        }]
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "text": f"String A: {string_a}\nString B: {string_b}"
                }
            ]
        }]
        
        # Using Claude 3 Haiku for fast, cheap JSON generation
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig={"temperature": 0.0}
        )
        
        output_text = response['output']['message']['content'][0]['text']
        
        # Parse JSON from response
        try:
            result = json.loads(output_text.strip())
            # Ensure the expected keys exist
            if "is_similar" in result and "string_a_normalized" in result and "string_b_normalized" in result:
                return result
        except json.JSONDecodeError:
            logger.warning(f"Bedrock returned invalid JSON: {output_text}")
            
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Bedrock API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in bedrock_preprocessor: {e}")

    # Fallback if Bedrock fails or credentials are not set
    return {
        "is_similar": True,
        "string_a_normalized": string_a,
        "string_b_normalized": string_b
    }

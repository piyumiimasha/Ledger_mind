"""
LLM and Embedding Factory Functions
Centralized factory functions to avoid code duplication across notebooks.
"""

import os
from typing import Dict, Any


class _SimpleLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _GroqClientWrapper:
    """Minimal invoke-compatible wrapper used when LangChain import path fails."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout: int):
        from groq import Groq

        self._client = Groq(api_key=api_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt: str):
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = completion.choices[0].message.content or ""
        return _SimpleLLMResponse(content)


# ============================================================================
# LangChain LLM Factory
# ============================================================================

def get_llm(config: Dict[str, Any]):
    """
    Return a LangChain-compatible chat model based on config.
    
    Args:
        config: Configuration dictionary with llm_provider, llm_model, etc.
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If provider is unknown
    """
    provider = config["llm_provider"]
    
    if provider == "groq":
        model_name = config["llm_model"]
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config["request_timeout"],
            )
        except (OSError, ImportError):
            return _GroqClientWrapper(
                api_key=api_key,
                model=model_name,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config["request_timeout"],
            )
    else:
        raise ValueError(f"Unknown llm_provider: {provider}")


# ============================================================================
# LangChain Embeddings Factory
# ============================================================================

def get_text_embeddings(config: Dict[str, Any]):
    """
    Return LangChain embeddings based on config.
    
    Args:
        config: Configuration dictionary with text_emb_provider, text_emb_model, etc.
        
    Returns:
        LangChain embeddings instance
        
    Raises:
        ValueError: If provider is unknown
    """
    provider = config["text_emb_provider"]
    model_name = config["text_emb_model"]
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    
    elif provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=model_name)
    
    elif provider == "sbert":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_kwargs = {"device": "cpu"} #cuda or mps
        
        if config.get("normalize_embeddings", True):
            encode_kwargs = {"normalize_embeddings": True}
        else:
            encode_kwargs = {}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    else:
        raise ValueError(f"Unknown text_emb_provider: {provider}")


def get_clip_model(config: Dict[str, Any]):
    """
    Return SentenceTransformer CLIP model for image+text embeddings.
    
    Args:
        config: Configuration dictionary with clip_model setting
        
    Returns:
        SentenceTransformer CLIP model instance
    """
    from sentence_transformers import SentenceTransformer
    clip_model_name = config.get("clip_model", "clip-ViT-B-32")
    return SentenceTransformer(clip_model_name)


# ============================================================================
# LlamaIndex LLM Factory
# ============================================================================

def get_llamaindex_llm(config: Dict[str, Any]):
    """
    Return LlamaIndex LLM based on config.
    
    Args:
        config: Configuration dictionary with llm_provider, llm_model, etc.
        
    Returns:
        LlamaIndex LLM instance
        
    Raises:
        ValueError: If provider is unknown
    """
    provider = config["llm_provider"]
        
    if provider == "groq":
        model_name = config["llm_model"]
        try:
            from llama_index_llms_groq import Groq
        except ImportError:
            from llama_index.llms.groq import Groq
        
        return Groq(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
        
    else:
        # Fallback to LangChain adapter
        model_name = config["llm_model"]
        from langchain_openai import ChatOpenAI
        from llama_index.core.llms import LangChainLLM
        lc_llm = ChatOpenAI(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
        return LangChainLLM(llm=lc_llm)


# ============================================================================
# LlamaIndex Embeddings Factory
# ============================================================================

def get_llamaindex_embeddings(config: Dict[str, Any]):
    """
    Return LlamaIndex embeddings based on config.
    
    Args:
        config: Configuration dictionary with text_emb_provider, text_emb_model, etc.
        
    Returns:
        LlamaIndex embeddings instance
        
    Raises:
        ValueError: If provider is unknown
    """
    provider = config["text_emb_provider"]
    model_name = config["text_emb_model"]
    
    if provider == "openai":
        try:
            from llama_index_embeddings_openai import OpenAIEmbedding
        except ImportError:
            from llama_index.embeddings.openai import OpenAIEmbedding
        
        return OpenAIEmbedding(model=model_name)
    
    elif provider == "cohere":
        try:
            from llama_index_embeddings_cohere import CohereEmbedding
        except ImportError:
            from llama_index.embeddings.cohere import CohereEmbedding
        
        return CohereEmbedding(model_name=model_name)
    
    elif provider == "sbert":
        try:
            from llama_index_embeddings_huggingface import HuggingFaceEmbedding
        except ImportError:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        return HuggingFaceEmbedding(
            model_name=model_name,
            device="cpu",
        )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# ============================================================================
# Utility Functions
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please ensure config.yaml exists in the project root."
        )
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def validate_api_keys(config: Dict[str, Any], verbose: bool = True) -> Dict[str, bool]:
    """
    Check which API keys are available in the environment.
    
    Args:
        config: Configuration dictionary
        verbose: If True, print warnings for missing keys
        
    Returns:
        Dictionary mapping key names to availability (True/False)
    """
    import warnings
    
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
    }
    
    availability = {}
    for key, value in api_keys.items():
        availability[key] = value is not None
        if verbose and not value:
            warnings.warn(f"⚠️  {key} not found in environment")
    
    return availability


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("✅ Config loaded:")
    
    # Display LLM info with OpenRouter-specific formatting
    if config['llm_provider'] == 'openrouter':
        openrouter_provider = config.get('openrouter_provider', 'openai')
        openrouter_model = config.get('openrouter_model', 'gpt-4o-mini')
        print(f"  LLM: {config['llm_provider']} ({openrouter_provider}/{openrouter_model})")
    else:
        print(f"  LLM: {config['llm_provider']} / {config['llm_model']}")
    
    print(f"  Embeddings: {config['text_emb_provider']} / {config['text_emb_model']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Artifacts: {config['artifacts_root']}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the factory functions.
    Run this script to test: python llm_services.py
    """
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load config
    config = load_config("config.yaml")
    
    # Print summary
    print_config_summary(config)
    print()
    
    # Validate API keys
    print("🔑 API Key Status:")
    availability = validate_api_keys(config, verbose=False)
    for key, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {key}")
    print()
    
    # Initialize LangChain LLM
    try:
        llm = get_llm(config)
        print(f"✅ LangChain LLM initialized: {config['llm_provider']}")
    except Exception as e:
        print(f"❌ LangChain LLM failed: {e}")
    
    # Initialize LangChain embeddings
    try:
        embeddings = get_text_embeddings(config)
        print(f"✅ LangChain embeddings initialized: {config['text_emb_provider']}")
    except Exception as e:
        print(f"❌ LangChain embeddings failed: {e}")
    
    print("\n🎉 All services initialized successfully!")


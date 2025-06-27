#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用本地LLM接口

支持多种开源模型的统一调用接口，用于记忆化vs推理能力评估
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional, Union
import logging
import gc

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLM:
    """本地LLM统一接口"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 max_length: int = 2048,
                 temperature: float = 0.1,
                 do_sample: bool = True):
        """
        初始化本地LLM
        
        Args:
            model_name: 模型名称或路径
            device: 设备 ('auto', 'cpu', 'cuda:0' 等)
            load_in_8bit: 是否使用8bit量化
            load_in_4bit: 是否使用4bit量化  
            max_length: 最大生成长度
            temperature: 温度参数
            do_sample: 是否采样
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        logger.info(f"Loading model: {model_name}")
        
        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # 配置量化
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.device = next(self.model.parameters()).device
            logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, 
                 prompt: Union[str, List[str]], 
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 do_sample: Optional[bool] = None,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 num_return_sequences: int = 1) -> Union[str, List[str]]:
        """
        生成文本
        
        Args:
            prompt: 输入提示(支持单个字符串或字符串列表)
            max_new_tokens: 最大新生成token数
            temperature: 温度参数
            do_sample: 是否采样
            top_p: top-p采样参数
            top_k: top-k采样参数
            num_return_sequences: 返回序列数量
            
        Returns:
            生成的文本
        """
        # 使用默认参数
        if max_new_tokens is None:
            max_new_tokens = 512
        if temperature is None:
            temperature = self.temperature
        if do_sample is None:
            do_sample = self.do_sample
            
        # 处理输入
        is_batch = isinstance(prompt, list)
        if not is_batch:
            prompt = [prompt]
        
        # 编码输入
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # 解码输出
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            
            generated_texts = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 返回结果
            if is_batch:
                return generated_texts
            else:
                return generated_texts[0]
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "" if not is_batch else [""] * len(prompt)
    
    def query(self, prompt: str, **kwargs) -> str:
        """
        简单查询接口（兼容现有代码）
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        return self.generate(prompt, **kwargs)
    
    def cleanup(self):
        """清理GPU内存"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

# 预定义的模型配置
SUPPORTED_MODELS = {
    "qwen-0.5b": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "load_in_4bit": False,
        "max_length": 2048
    },
    "qwen-1.5b": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct", 
        "load_in_4bit": False,
        "max_length": 2048
    },
    "qwen-3b": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "load_in_4bit": True,
        "max_length": 2048
    },
    "llama-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "load_in_4bit": False,
        "max_length": 2048
    },
    "llama-3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "load_in_4bit": True,
        "max_length": 2048
    },
    "phi-3.5": {
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "load_in_4bit": True,
        "max_length": 2048
    },
    "gemma-2b": {
        "model_name": "google/gemma-2-2b-it",
        "load_in_4bit": False,
        "max_length": 2048
    }
}

def load_model(model_key: str = "qwen-0.5b", **kwargs) -> LocalLLM:
    """
    加载预定义模型
    
    Args:
        model_key: 模型键值
        **kwargs: 额外参数
        
    Returns:
        LocalLLM实例
    """
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_key}. Available: {list(SUPPORTED_MODELS.keys())}")
    
    config = SUPPORTED_MODELS[model_key].copy()
    config.update(kwargs)
    
    return LocalLLM(**config)

# 测试函数
def test_model(model_key: str = "qwen-0.5b"):
    """测试模型加载和生成"""
    print(f"Testing model: {model_key}")
    
    try:
        # 加载模型
        llm = load_model(model_key)
        
        # 测试生成
        test_prompts = [
            "What is 2+2?",
            "Solve this logic puzzle: If A is true, then B is false. A is true. What is B?",
            "Complete the sequence: 2, 4, 6, 8, ?"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = llm.query(prompt, max_new_tokens=100)
            print(f"Response: {response}")
        
        # 清理
        llm.cleanup()
        print(f"\n✅ Model {model_key} test completed successfully!")
        
    except Exception as e:
        print(f"❌ Model {model_key} test failed: {e}")

if __name__ == "__main__":
    # 显示支持的模型
    print("Supported models:")
    for key, config in SUPPORTED_MODELS.items():
        print(f"  {key}: {config['model_name']}")
    
    # 测试最小模型
    print("\n" + "="*50)
    test_model("qwen-0.5b") 
import requests
from PIL import Image

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.schema import OutputParserException

import json
from tqdm import tqdm
tqdm.pandas()

import google.generativeai as genai
from dotenv import load_dotenv
import os
from io import BytesIO

API_KEY = "AIzaSyAZTNYe-qulKkpJRmmgwsVdVYrdCWyycvk"
genai.configure(api_key=API_KEY)


def gemini_call(sample_1,pack_1):
    image_path = r"C:\Users\liyenga\OneDrive - Veritiv Corp\Documents\Veritiv Projects\Image_description\images\Pic1.png"

    image_sample = Image.open(image_path)
    title_sample= "Scotch® 311+ Clear High Tack Box Sealing Tape (48 mm. x 914 m., 6 Rolls/Case)"
    # Example usage
    
            
    res=[]
    for image_url, title_string in zip(sample_1,pack_1):
    
        response = requests.get(image_url, verify=False)
        image = Image.open(BytesIO(response.content))

        prompt = f"""Write a product description based on the image {image} and Title {title_string} provided. Please follow the example below and use the same tone, length, and details in the example below
                example: for the given image {image_sample} with title- {title_sample}
                ,here's a sample of output description
                Description: Seal your packages securely with Scotch® 311+ Clear High Tack Box Sealing Tape. Designed for high-performance packaging needs, 
                this tape offers excellent adhesion and durability. Its clear construction ensures a clean, professional appearance, making it ideal for 
                shipping, storage, and general sealing applications. Depend on Scotch® 311+ for reliable, strong seals that stand up to the demands of your busy environment.

                """
        model = genai.GenerativeModel("gemini-1.5-flash")
        result_1 = model.generate_content([prompt, image]).text
    
    
        #display(image.resize((480, 600)))
        res.append(result_1)
    return res

# azure openai params
load_dotenv('azure_openai_3.5_turbo.env')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_VERSION')
deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')
model_name = os.getenv('AZURE_OPENAI_MODEL_NAME')

def output_llm(chain, image_path,title_string, image_sample,title_sample):
    
    chain_result = chain.invoke({"image_path": image_path,"title_string": title_string, "image_sample": image_sample,"title_sample": title_sample})
    return chain_result.get('Description', None)

def azure_call(sample_1,pack_1):

    prompt_template_2 = """
    Write a product description based on the image {image_path} and Title {title_string} provided. The marketing description has to mention the uses and benefits of the product as to be explained to a customer in about 500 characters
    Please follow the example below and use the same tone, length, and details in the example below
                example: for the given Scotch tape {image_sample} with title- {title_sample}
                ,here's a sample of output description
                OUTPUT - Description: Seal your packages securely with Scotch® 311+ Clear High Tack Box Sealing Tape. Designed for high-performance packaging needs, this tape offers excellent adhesion and durability. Its clear construction ensures a clean, professional appearance, making it ideal for shipping, storage, and general sealing applications. Depend on Scotch® 311+ for reliable, strong seals that stand up to the demands of your busy environment.

    {format_instructions}
    """

    attributes_schema = ResponseSchema(name="Description",
                                   description="Description of image and title given",
                                   type="str")
    output_parser = StructuredOutputParser.from_response_schemas([attributes_schema])

    format_instructions = output_parser.get_format_instructions()

    prompt_2 = PromptTemplate(template=prompt_template_2,
                        input_variables=["image_path","title_string", "image_sample","title_sample"],
                        partial_variables={"format_instructions": format_instructions})


    model_2 = AzureChatOpenAI(azure_endpoint=azure_endpoint, api_key=api_key,
                        api_version=api_version, deployment_name=deployment_name,
                        model_name=model_name, 
                        temperature=0) 

    chain = prompt_2 | model_2 | output_parser

    image_sample = Image.open(requests.get("https://105103-veritiv-prod.s3.amazonaws.com/Damroot/xLg/10071/IMAGE_20044788_Pic%201.png", verify=False, stream=True).raw).convert('RGB')  
    title_sample= "Scotch® 311+ Clear High Tack Box Sealing Tape (48 mm. x 914 m., 6 Rolls/Case)"
    
    res2=[]
    # Example usage
    for image_url, title_string in zip(sample_1,pack_1):
    
        response = requests.get(image_url, verify=False)
        image = Image.open(BytesIO(response.content))
        
        result = output_llm(chain, image,title_string, image_sample,title_sample)
    
        res2.append(result)
    return res2

def blip_call(sample_ove,sam_tit_ove):

    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    import torch

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image_sample = Image.open(requests.get("https://105103-veritiv-prod.s3.amazonaws.com/Damroot/xLg/10071/IMAGE_20044788_Pic%201.png", verify=False, stream=True).raw).convert('RGB')  
    title_sample= "Scotch® 311+ Clear High Tack Box Sealing Tape (48 mm. x 914 m., 6 Rolls/Case)"
    res= []
    for image_url, title_string in zip(sample_ove,sam_tit_ove):
    
        response = requests.get(image_url, verify=False)
        image = Image.open(BytesIO(response.content))

        prompt = f"""Write a product description based on the image {image} and Title {title_string} provided. Please follow the example below and use the same tone, length, and details in the example below
                example: for the given image {image_sample} with title- {title_sample}
                ,here's a sample of output description
                Description: Seal your packages securely with Scotch® 311+ Clear High Tack Box Sealing Tape. Designed for high-performance packaging needs, 
                this tape offers excellent adhesion and durability. Its clear construction ensures a clean, professional appearance, making it ideal for 
                shipping, storage, and general sealing applications. Depend on Scotch® 311+ for reliable, strong seals that stand up to the demands of your busy environment.

                """
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
        description = generated_text.split("Description:")[1].strip()
        #display(image.resize((480, 600)))
        res.append(description)

    return res

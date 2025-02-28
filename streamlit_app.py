import json
import os
import platform
import sys
import traceback
import uuid
import math
from urllib.parse import urlparse, unquote
from io import StringIO, BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import streamlit as st
import pandas as pd
import boto3
import base64
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pydantic import BaseModel

# Remove hard-coded API key import
# from config.sys_config import OPENAI_API_KEY

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="File Prepare",
    layout="wide"
)

# -------------------------
# Credentials Input Section
# -------------------------
st.sidebar.header("Enter Your Credentials")

# Initialize session state variables if not already set
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "aws_access_key_id" not in st.session_state:
    st.session_state["aws_access_key_id"] = ""
if "aws_secret_access_key" not in st.session_state:
    st.session_state["aws_secret_access_key"] = ""
if "aws_region" not in st.session_state:
    st.session_state["aws_region"] = "us-east-1"
if "S3_BUCKET_NAME" not in st.session_state:
    st.session_state["S3_BUCKET_NAME"] = ""

# Option 1: Manual entry via text inputs
st.session_state["openai_api_key"] = st.sidebar.text_input(
    "OpenAI API Key:", value=st.session_state["openai_api_key"], type="password"
)
st.session_state["aws_access_key_id"] = st.sidebar.text_input(
    "AWS Access Key ID:", value=st.session_state["aws_access_key_id"], type="password"
)
st.session_state["aws_secret_access_key"] = st.sidebar.text_input(
    "AWS Secret Access Key:", value=st.session_state["aws_secret_access_key"], type="password"
)
st.session_state["aws_region"] = st.sidebar.text_input(
    "AWS Region (default is us-east-1):", value=st.session_state["aws_region"]
)
st.session_state["S3_BUCKET_NAME"] = st.sidebar.text_input(
    "S3 Bucket Name:", value=st.session_state["S3_BUCKET_NAME"]
)

# Option 2: Import credentials via file upload
creds_file = st.sidebar.file_uploader("Import Credentials File", type=["txt"], key="creds_file")
if creds_file is not None:
    try:
        creds_text = creds_file.read().decode("utf-8")
        for line in creds_text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "aws_access_key_id":
                    st.session_state["aws_access_key_id"] = value
                    st.sidebar.write(f"Imported {key}")
                elif key == "aws_secret_access_key":
                    st.session_state["aws_secret_access_key"] = value
                    st.sidebar.write(f"Imported {key}")
                elif key == "openai_api_key":
                    st.session_state["openai_api_key"] = value
                    st.sidebar.write(f"Imported {key}")
                elif key == "aws_region":
                    st.session_state["aws_region"] = value
                    st.sidebar.write(f"Imported {key}")
                elif key == "S3_BUCKET_NAME":
                    st.session_state["S3_BUCKET_NAME"] = value
                    st.sidebar.write(f"Imported {key}")
        st.sidebar.success("Credentials imported successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading credentials file: {e}")

# Validate credentials before proceeding
if (not st.session_state["openai_api_key"] or
    not st.session_state["aws_access_key_id"] or
    not st.session_state["aws_secret_access_key"] or
    not st.session_state["S3_BUCKET_NAME"]):
    st.warning("Please enter your OpenAI API key and AWS credentials in the sidebar to use the app.")
    st.stop()


# -------------------------
# Initialize Clients Dynamically
# -------------------------
from openai import OpenAI  # Import after credentials are provided

openai_client = OpenAI(api_key=st.session_state["openai_api_key"])
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.session_state["aws_access_key_id"],
    aws_secret_access_key=st.session_state["aws_secret_access_key"],
    region_name=st.session_state["aws_region"]
)

# -------------------------
# Utility Functions
# -------------------------
def parse_s3_url(s3_url):
    """
    Parses the given S3 URL and returns the bucket name and key.
    Supports virtual-hosted-style and path-style URLs.
    """
    parsed_url = urlparse(s3_url)
    hostname = parsed_url.netloc
    path = parsed_url.path.lstrip('/')
    
    # Handle URL encoding in the path
    key = unquote(path)
    
    # Determine the bucket name based on the hostname
    if hostname.endswith('amazonaws.com'):
        # Virtual-hosted-style URL
        parts = hostname.split('.')
        if parts[0] == 's3':
            # Path-style URL
            bucket_name = parts[2]
            key = f"{parts[3]}/{key}" if len(parts) > 3 else key
        else:
            # Virtual-hosted-style URL
            bucket_name = parts[0]
    else:
        bucket_name = hostname

    return bucket_name, key

# Create folder in S3 bucket
def create_folder_in_s3_bucket(s3_url):
    try:
        bucket_name, folder_key = parse_s3_url(s3_url)
        if not folder_key.endswith('/'):
            folder_key += '/'
        s3_client.put_object(Bucket=bucket_name, Key=folder_key)
        return True
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Credentials error:", e)
        return False
    except Exception as e:
        print("An error occurred:", e)
        return False
    
def create_or_update_file_in_s3(s3_url, file_content, file_type="text"):
    """
    Creates or updates a file in S3.
    """
    try:
        bucket_name, file_key = parse_s3_url(s3_url)
        if file_type == "text":
            if not isinstance(file_content, str):
                raise ValueError("For text, file_content must be a string.")
            body = file_content
        elif file_type == "json":
            if isinstance(file_content, pd.DataFrame):
                body = file_content.to_json(orient='records')
            elif isinstance(file_content, dict):
                def serialize_value(value):
                    if isinstance(value, pd.DataFrame):
                        return value.to_dict(orient='records')
                    elif isinstance(value, list):
                        return [serialize_value(item) for item in value]
                    elif isinstance(value, dict):
                        return {k: serialize_value(v) for k, v in value.items()}
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        return value
                    else:
                        return str(value)
                
                file_content = {k: serialize_value(v) for k, v in file_content.items()}
                body = json.dumps(file_content, indent=4)
            elif isinstance(file_content, list):
                def serialize_list(value):
                    if isinstance(value, pd.DataFrame):
                        return value.to_dict(orient='records')
                    elif isinstance(value, dict):
                        return {k: serialize_value(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        return [serialize_value(item) for item in value]
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        return value
                    else:
                        return str(value)
                
                body = json.dumps([serialize_list(item) for item in file_content], indent=4)
            else:
                raise ValueError("For JSON, file_content must be a dict, list, or string.")
        elif file_type == "csv":
            if not isinstance(file_content, pd.DataFrame):
                raise ValueError("For CSV, file_content must be a pandas DataFrame.")
            body = file_content.to_csv(index=False)
        elif file_type == "image":
            if not isinstance(file_content, Image.Image):
                raise ValueError("For image, file_content must be a PIL Image object.")
            img_byte_arr = BytesIO()
            if file_key.endswith(".png"):
                file_content.save(img_byte_arr, format='PNG')
            elif file_key.endswith(".jpg") or file_key.endswith(".jpeg"):
                file_content.save(img_byte_arr, format='JPEG')
            elif file_key.endswith(".bmp"):
                file_content.save(img_byte_arr, format='BMP')
            elif file_key.endswith(".tiff"):
                file_content.save(img_byte_arr, format='TIFF')
            elif file_key.endswith(".gif"):
                file_content.save(img_byte_arr, format='GIF')
            else:
                raise ValueError("Unsupported image format.")
            body = img_byte_arr.getvalue()
        else:
            raise ValueError("Invalid file type. Options are 'text', 'json', 'csv', 'image'.")

        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=body)
        return s3_url

    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Credentials error:", e)
        return None
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

TRANSLATOR = """
You are a professional translator specializing in creating high-quality, listing-ready English titles and descriptions for e-commerce platforms. You will translate item titles and descriptions from Chinese to English, ensuring they are clear, accurate, and optimized for online listings. The translation should meet the following criteria:

- Accuracy: Retain all essential information from the original text, ensuring no details are omitted or altered.
- Readability: Use natural, fluent English suitable for online shoppers.
- Formatting: Ensure the text is concise and formatted appropriately for e-commerce platforms.
   - Use numerals for measurements, dimensions, and quantities (e.g., "6.5" instead of "six and a half").
   - Use <br> tags to match the original's line breaks.
   - Avoid unnecessary special characters or formatting beyond what is required.
- Style: Adapt the tone to match the item type:
   - Sneakers: Use trendy, casual language that appeals to fashion-conscious buyers.
   - Vintage Furniture: Highlight elegance, uniqueness, and historical charm.
   - Home Decor: Emphasize aesthetic appeal, functionality, and style.
- Localization: Replace culturally specific references with equivalent terms or explanations familiar to an English-speaking audience, where appropriate.
- Do Not Include: Avoid references to warranties, guarantees, or promotional terms unless explicitly stated in the original text.

For example:

Chinese: 一双经典款白色运动鞋，适合日常穿着。
English: A pair of classic white sneakers, perfect for everyday wear.
"""

class TranslationResult(BaseModel):
    translated_text: str


# -------------------------
# OpenAI & Boto3 Client Configuration
# (Already initialized above with user credentials)
# -------------------------
S3_BUCKET = st.session_state["S3_BUCKET_NAME"]
s3_bucket_parent_folder = S3_BUCKET.split("/")[0]
s3_bucket_child_folder = S3_BUCKET.split("/")[1]
MODEL_1 = "gpt-4o-mini-2024-07-18"
MAX_TOKENS = 16384
TEMPERATURE = 0.7
RETRY_ATTEMPTS = 3

# -------------------------
# Initialize Session State Variables
# -------------------------
if "file_name" not in st.session_state:
    st.session_state["file_name"] = ""

if "sku_prefix" not in st.session_state:
    st.session_state["sku_prefix"] = ""

if "generated_sku" not in st.session_state:
    st.session_state["generated_sku"] = ""

if "translated_title" not in st.session_state:
    st.session_state["translated_title"] = ""

if "translated_description" not in st.session_state:
    st.session_state["translated_description"] = ""

if "original_price" not in st.session_state:
    st.session_state["original_price"] = 0

if "percentage" not in st.session_state:
    st.session_state["percentage"] = 100

if "discount_price" not in st.session_state:
    st.session_state["discount_price"] = 0

if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = []

if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = pd.DataFrame()

desired_order = [
    "Sku", "Title", "Description", "original_price", "discounted_price",
    "Preview_ImageFile.1", "ImageFile.1", "Preview_ImageFile.2", "ImageFile.2",
    "Preview_ImageFile.3", "ImageFile.3", "Preview_ImageFile.4", "ImageFile.4",
    "Preview_ImageFile.5", "ImageFile.5", "Preview_ImageFile.6", "ImageFile.6",
    "Preview_ImageFile.7", "ImageFile.7", "Preview_ImageFile.8", "ImageFile.8",
    "Preview_ImageFile.9", "ImageFile.9", "Preview_ImageFile.10", "ImageFile.10"
]

saved_order = [
    "Sku", "Title", "Description", "original_price", "discounted_price", 
    "ImageFile.1", "ImageFile.2", "ImageFile.3", "ImageFile.4",
    "ImageFile.5", "ImageFile.6", "ImageFile.7", "ImageFile.8", 
    "ImageFile.9", "ImageFile.10"
]

def generate_sku(prefix):
    return f"{prefix}{uuid.uuid4().hex[:7]}"

def translate_text(text, target_language="en"):
    response = openai_client.beta.chat.completions.parse(
        model=MODEL_1,
        messages=[
            {"role": "system", "content": TRANSLATOR},
            {"role": "user", "content": f"Translate this to {target_language}: {text}"}
        ],
        response_format=TranslationResult,
    )
    return response.choices[0].message.parsed.translated_text

def upload_to_s3(image_name, image_content, folder_name):
    try:
        create_folder_in_s3_bucket(f"https://{s3_bucket_parent_folder}.s3.{st.session_state['aws_region']}.amazonaws.com/{s3_bucket_child_folder}/{folder_name}")
        file_s3_url = f"https://{s3_bucket_parent_folder}.s3.{st.session_state['aws_region']}.amazonaws.com/{s3_bucket_child_folder}/{folder_name}/{image_name}"
        s3_url = create_or_update_file_in_s3(file_s3_url, image_content, "image")
        return s3_url
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")
        return None

def generate_presigned_url_from_https(s3_https_url, expiration=604800):
    """
    Generate a pre-signed URL from an HTTPS S3 URL.
    """
    try:
        parsed_url = urlparse(s3_https_url)
        if parsed_url.netloc.endswith(".amazonaws.com"):
            bucket_name = parsed_url.netloc.split(".")[0]
            object_key = parsed_url.path.lstrip("/")
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return str(url)
        else:
            st.error("Invalid S3 URL format.")
            return ""
    except Exception as e:
        st.error(f"Error generating pre-signed URL: {e}")
        return ""

def get_downloads_folder():
    if platform.system() == "Windows":
        return os.path.join(os.path.expanduser("~"), "Downloads")
    elif platform.system() == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Downloads")
    else:
        return os.path.expanduser("~")
    
# -------------------------
# Streamlit App Main Section
# -------------------------
st.title("CSV File Preparation Tool")

# Section 0(Optional): Upload a CSV to Replace Table
st.subheader("Upload CSV to Replace Table")
st.warning("WARNING: Refresh the webpage BEFORE uploading. Consider save your current table first.")

if "csv_uploaded" not in st.session_state:
    st.session_state["csv_uploaded"] = False

uploaded_csv = st.file_uploader("Upload CSV File:", type="csv", key="uploaded_csv")
if uploaded_csv and not st.session_state["csv_uploaded"]:
    try:
        uploaded_df = pd.read_csv(uploaded_csv)
        st.session_state["dataframe"] = uploaded_df
        current_file_name = st.session_state["file_name"]
        if current_file_name != uploaded_csv.name.split(".")[0]:
            st.session_state["file_name"] = uploaded_csv.name.split(".")[0]
        st.session_state["csv_uploaded"] = True
        st.success("CSV uploaded, can view it in the table view.")

        for i in range(1, 11):
            if f"Preview_ImageFile.{i}" not in st.session_state["dataframe"].columns:
                st.session_state["dataframe"][f"Preview_ImageFile.{i}"] = ""
        for index, row in st.session_state["dataframe"].iterrows():
            for i in range(1, 11):
                image_url = row[f"ImageFile.{i}"]
                if pd.notna(image_url):
                    image_preview_url = generate_presigned_url_from_https(image_url)
                    st.session_state["dataframe"].at[index, f"Preview_ImageFile.{i}"] = image_preview_url
                    st.session_state["dataframe"] = st.session_state["dataframe"][desired_order]
        
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")

st.warning("WARNING: Click this button before re-uploading. Consider save your current table first.")
if st.button("Reset CSV Upload"):
    st.session_state["csv_uploaded"] = False
    st.success("CSV upload flag reset. You can now upload a new file.")

st.markdown("<hr>", unsafe_allow_html=True)

# Section 1: User Input
st.header("Input Section")
file_name = st.text_input("Enter File Name for CSV (no need to add .csv):",
                          value=st.session_state["file_name"],
                          key="file_name_input")
st.session_state["file_name"] = file_name

st.markdown("<hr>", unsafe_allow_html=True)

sku_prefix = st.text_input("Enter SKU Prefix:", value=st.session_state["sku_prefix"])
st.session_state["sku_prefix"] = sku_prefix
if st.button("Generate SKU"):
    sku = generate_sku(sku_prefix)
    st.session_state["generated_sku"] = sku

st.text_area("Generated Sku (Editable):", value=st.session_state.get("generated_sku", ""), 
             key="generated_sku_editable", height=70)
st.session_state["generated_sku"] = st.session_state["generated_sku_editable"]

st.markdown("<hr>", unsafe_allow_html=True)

title = st.text_input("Enter Title:")
if st.button("Translate Title"):
    translated_title = translate_text(title)
    st.session_state["translated_title"] = translated_title
st.text_area("Translated Title (Editable):", value=st.session_state.get("translated_title", ""), 
             key="translated_title_editable", height=100)
st.session_state["translated_title"] = st.session_state["translated_title_editable"]

st.markdown("<hr>", unsafe_allow_html=True)

description = st.text_area("Enter Description:")
if st.button("Translate Description"):
    translated_description = translate_text(description)
    st.session_state["translated_description"] = translated_description
st.text_area("Translated Description (Editable):", value=st.session_state.get("translated_description", ""),
             key="translated_description_editable", height=200)
st.session_state["translated_description"] = st.session_state["translated_description_editable"]

st.markdown("<hr>", unsafe_allow_html=True)

original_price = st.number_input("Enter Original Price in USD:",
                                 min_value=0.0,
                                 step=0.5,
                                 value=float(st.session_state.get("original_price", 0.0)))
st.session_state["original_price"] = original_price

percentage = st.number_input("Set Discount Percentage (%):",
                             min_value=0.0,
                             max_value=100.0,
                             step=0.1,
                             value=float(st.session_state.get("percentage", 100.0)))
st.session_state["percentage"] = percentage

calculated_discount_price = math.ceil(original_price * (percentage / 100.0))
st.session_state["discount_price"] = calculated_discount_price

discount_price = st.number_input("Auction House Price in USD (Editable):",
                                      min_value=0.0,
                                      step=0.5,
                                      value=float(st.session_state.get("discount_price", calculated_discount_price)))
if discount_price != calculated_discount_price:
    st.warning("The Auction House Price has been manually modified. Ensure the value is correct.")
st.session_state["discount_price"] = discount_price

st.markdown("<hr>", unsafe_allow_html=True)

# Section 2: Image Upload and CSV Generation
st.header("Image Upload")
st.session_state["uploaded_images"] = []  



def image_to_base64(image_bytes, file_type):
    """
    Convert image bytes to a Base64 data URL.
    """
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    # Use the proper MIME type (note: jpg is mapped to jpeg)
    mime = f"image/{file_type}" if file_type.lower() != "jpg" else "image/jpeg"
    return f"data:{mime};base64,{base64_str}"

for i in range(10):
    uploaded_file = st.file_uploader(
        f"Upload Image {i + 1}:", type=["jpg", "jpeg", "png"], key=f"image_{i}"
    )
    if uploaded_file:
        file_type = uploaded_file.type.split("/")[1]
        file_key = f"{st.session_state['generated_sku']}-{i + 1}.{file_type}"
        file_bytes = uploaded_file.getvalue()
        # st.write(f"Image {i+1}")
        try:
            # For preview, convert the raw bytes to a Base64 string and build an HTML image tag.
            base64_image = image_to_base64(file_bytes, file_type)
            html_code = f'<img src="{base64_image}" alt="Uploaded Image {i+1}" width="200">'
            st.markdown(html_code, unsafe_allow_html=True)
            
            # For upload, open the image using PIL without converting it.
            # (This avoids the problematic conversion step.)
            image_obj = Image.open(BytesIO(file_bytes))
            try:
                image_content = image_obj.copy()  # Copy the image to get a proper PIL Image object.
            except Exception as e:
                st.warning(f"Image copy failed: {e}. Using original object.")
                image_content = image_obj

            s3_url = upload_to_s3(file_key, image_content, st.session_state["file_name"])
            if s3_url:
                st.session_state["uploaded_images"].append(s3_url)
            else:
                st.session_state["uploaded_images"].append("")
        except Exception as e:
            st.error(f"Error processing image {i + 1}: {e}\n{traceback.format_exc()}")
            st.session_state["uploaded_images"].append("")
    else:
        st.session_state["uploaded_images"].append("")


st.markdown("<hr>", unsafe_allow_html=True)

# Section 3: Upload Data
st.header("Data Preview")
data = {
    "Sku": st.session_state["generated_sku"],
    "Title": st.session_state["translated_title"],
    "Description": st.session_state["translated_description"],
    "original_price": st.session_state["original_price"],
    "discounted_price": st.session_state["discount_price"],
}

column_config = {}
for i in range(0, len(st.session_state["uploaded_images"])):
    data[f"Preview_ImageFile.{i + 1}"] = str(generate_presigned_url_from_https(st.session_state["uploaded_images"][i]) if st.session_state["uploaded_images"][i] else "")
    data[f"ImageFile.{i + 1}"] = str(st.session_state["uploaded_images"][i])
    column_config[f"Preview_ImageFile.{i+1}"] = st.column_config.ImageColumn(f"Preview Image {i+1}")

if data["Sku"] and data["Title"] and data["Description"]:
    new_single_row_df = st.data_editor(pd.DataFrame([data]), 
                                         use_container_width=True,
                                         column_config=column_config,
                                         hide_index=True,
                                         key="data_editor")
else:
    st.error("Please fill in all required fields.")

st.markdown("<hr>", unsafe_allow_html=True)

# Section 4: Dynamic Table View
st.header("Table View")
if st.button("Click to refresh and view data in Table (with image previews attached)"):
    # Check that required fields are non-empty after stripping whitespace
    if not (data.get("Sku", "").strip() and data.get("Title", "").strip() and data.get("Description", "").strip()):
        st.error("Required fields (Sku, Title, Description) must not be empty.")
    else:
        if st.session_state["file_name"] == "":
            st.error("Please scroll up and enter a file name for the CSV.")
        else:
            if isinstance(st.session_state.dataframe, pd.DataFrame) and not st.session_state.dataframe.empty:
                if data["Sku"] in st.session_state.dataframe["Sku"].values:
                    st.session_state.dataframe.loc[
                        st.session_state.dataframe["Sku"] == data["Sku"], :
                    ] = pd.DataFrame([data]).astype(st.session_state.dataframe.dtypes.to_dict())
                else:
                    st.session_state["dataframe"] = pd.concat(
                        [st.session_state.dataframe, pd.DataFrame([data])], ignore_index=True
                    )
            else:
                st.session_state["dataframe"] = pd.DataFrame([data])

    # check and remove empty rows
    if isinstance(st.session_state.dataframe, pd.DataFrame) and not st.session_state.dataframe.empty:
        st.session_state.dataframe = st.session_state.dataframe.dropna(how='all')


# Define column configuration with ImageColumn for the preview image columns
column_config_table = {
    "Sku": st.column_config.TextColumn("SKU"),
    "Title": st.column_config.TextColumn("Title"),
    "Description": st.column_config.TextColumn("Description"),
    "original_price": st.column_config.NumberColumn("Original Price"),
    "discounted_price": st.column_config.NumberColumn("Discounted Price"),
}

# Dynamically add preview image columns configuration
for i in range(0, len(st.session_state["uploaded_images"])):
    preview_col = f"Preview_ImageFile.{i+1}"
    column_config_table[preview_col] = st.column_config.ImageColumn(f"Preview Image {i+1}")

updated_dataframe = st.data_editor(
    st.session_state["dataframe"], 
    num_rows="dynamic", 
    use_container_width=True, 
    column_config=column_config_table
)


# actually update the dataframe
# **Ensure session state updates with the latest changes**
# st.session_state["dataframe"] = updated_dataframe
print("st.session_state['dataframe'] after display:", st.session_state["dataframe"])


# Section 5: Provide Download Option
st.subheader("Save Table")

# Summary of the data
st.write("Table Summary:")
if not st.session_state["dataframe"].empty:
    num_rows = len(st.session_state["dataframe"])
    st.write(f"**Number of Rows**: {num_rows}")
    st.write(f"**CSV File Name:** {st.session_state['file_name']}.csv")
else:
    st.write("The table is currently empty.")

if not st.session_state["dataframe"].empty and st.session_state["file_name"]:
    try:
        latest_dataframe = st.session_state["dataframe"].copy()
        latest_dataframe = latest_dataframe[saved_order]
        latest_dataframe = latest_dataframe[latest_dataframe["Sku"].notna()]
        latest_dataframe = latest_dataframe.convert_dtypes()

        csv_buffer = BytesIO()
        latest_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_bytes = csv_buffer.getvalue()

        file_key = f"{st.session_state['file_name']}.csv"

        st.download_button(
            label="Download Table Locally",
            data=csv_bytes,
            file_name=file_key,
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing table: {e}")
else:
    st.warning("Please enter a file name and ensure the table is not empty before saving or downloading.")

st.markdown("<hr>", unsafe_allow_html=True)

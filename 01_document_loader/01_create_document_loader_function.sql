use role sysadmin;
create database if not exists sandbox;
use schema sandbox.public;

create stage file;
alter stage file set directory = (enable=TRUE);

CREATE OR REPLACE FUNCTION load_pdf(file_path string)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python','langchain', 'langchain-community', 'pypdf')
HANDLER = 'run'
AS
$$
import tempfile

from langchain.document_loaders import PyPDFLoader
from snowflake.snowpark.files import SnowflakeFile

def run(file_path):
    with SnowflakeFile.open(file_path, 'rb') as f:
        pdf_bytes = f.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    pdf_text = "\n".join(doc.page_content for doc in documents)
    return pdf_text
$$;
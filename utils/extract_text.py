"""
Text extraction utilities for CV files.
Supports PDF, DOCX, and TXT formats.
"""

import io
from typing import Union, BinaryIO
import pdfplumber
import docx2txt


def extract_text(file: Union[BinaryIO, str]) -> str:
    """
    Extract text from uploaded file based on file type.
    
    Args:
        file: File object (UploadedFile from Streamlit) or file path
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If file type is not supported
    """
    # Get file name and extension
    if hasattr(file, 'name'):
        filename = file.name
    else:
        filename = str(file)
    
    file_extension = filename.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        return extract_pdf(file)
    elif file_extension in ['docx', 'doc']:
        return extract_docx(file)
    elif file_extension == 'txt':
        return extract_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Please upload PDF, DOCX, or TXT file.")


def extract_pdf(file: Union[BinaryIO, str]) -> str:
    """
    Extract text from PDF file using pdfplumber.
    
    Args:
        file: PDF file object or path
        
    Returns:
        str: Extracted text from all pages
    """
    text = ""
    
    try:
        # Handle file object vs file path
        if hasattr(file, 'read'):
            # It's a file object
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            # It's a file path
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    return text.strip()


def extract_docx(file: Union[BinaryIO, str]) -> str:
    """
    Extract text from DOCX file using docx2txt.
    
    Args:
        file: DOCX file object or path
        
    Returns:
        str: Extracted text
    """
    try:
        # Handle file object vs file path
        if hasattr(file, 'read'):
            # It's a file object - need to pass path or file-like object
            text = docx2txt.process(file)
        else:
            # It's a file path
            text = docx2txt.process(file)
            
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")


def extract_txt(file: Union[BinaryIO, str]) -> str:
    """
    Extract text from TXT file.
    
    Args:
        file: TXT file object or path
        
    Returns:
        str: File content
    """
    try:
        # Handle file object vs file path
        if hasattr(file, 'read'):
            # It's a file object
            if hasattr(file, 'getvalue'):
                # BytesIO or similar
                content = file.getvalue()
            else:
                content = file.read()
            
            # Decode if bytes
            if isinstance(content, bytes):
                text = content.decode('utf-8')
            else:
                text = content
        else:
            # It's a file path
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from TXT: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and special characters.
    
    Args:
        text: Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    # This is a simple version - can be enhanced based on needs
    
    return text.strip()

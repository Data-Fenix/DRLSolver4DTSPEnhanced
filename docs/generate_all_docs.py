"""
Main script to generate all documentation files (Markdown and Word)
Run this script to generate all documentation at once.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        import docx
        print("✓ python-docx already installed")
    except ImportError:
        print("Installing python-docx...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
        print("✓ python-docx installed")

def generate_markdown_files():
    """Generate markdown files (content provided separately)"""
    print("\n" + "="*60)
    print("Generating Markdown Files")
    print("="*60)
    print("Note: Markdown files should be created manually from the content")
    print("provided in the previous response.")
    print("Files to create:")
    print("  - DECODER_README.md")
    print("  - DECODER_DOCUMENTATION.md")

def generate_word_files():
    """Generate Word documents"""
    print("\n" + "="*60)
    print("Generating Word Documents")
    print("="*60)
    
    try:
        from generate_decoder_readme_word import create_decoder_readme_word
        create_decoder_readme_word()
    except Exception as e:
        print(f"Error generating README Word doc: {e}")
    
    try:
        from generate_decoder_documentation_word import create_decoder_documentation_word
        create_decoder_documentation_word()
    except Exception as e:
        print(f"Error generating Documentation Word doc: {e}")

def main():
    print("="*60)
    print("Decoder Documentation Generator")
    print("="*60)
    
    # Install requirements
    install_requirements()
    
    # Generate files
    generate_markdown_files()
    generate_word_files()
    
    print("\n" + "="*60)
    print("Documentation Generation Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ DECODER_README.docx")
    print("  ✓ DECODER_DOCUMENTATION.docx")
    print("\nNote: Create markdown files manually from provided content.")

if __name__ == '__main__':
    main()
"""
Utility script to convert files from UTF-16 encoding to UTF-8
"""
import os
import sys
import chardet

def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected {encoding} with {confidence:.2%} confidence for {file_path}")
        return encoding

def convert_file_encoding(file_path, source_encoding=None, target_encoding='utf-8'):
    """
    Convert a file from one encoding to another
    
    Args:
        file_path: Path to the file
        source_encoding: Original encoding (None for auto-detect)
        target_encoding: Target encoding (default: UTF-8)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Auto-detect encoding if not provided
        if source_encoding is None:
            source_encoding = detect_file_encoding(file_path)
        
        # Skip if already the target encoding
        if source_encoding.lower() == target_encoding.lower():
            print(f"File {file_path} is already in {target_encoding} encoding. Skipping.")
            return True
        
        # Read the file with source encoding
        with open(file_path, 'r', encoding=source_encoding, errors='replace') as file:
            content = file.read()
        
        # Write with target encoding
        with open(file_path, 'w', encoding=target_encoding) as file:
            file.write(content)
            
        print(f"Successfully converted {file_path} from {source_encoding} to {target_encoding}")
        return True
    
    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        return False

def convert_directory(directory_path, extensions=None, source_encoding=None, target_encoding='utf-8'):
    """
    Convert all files in a directory from one encoding to another
    
    Args:
        directory_path: Path to directory
        extensions: List of file extensions to convert (default: all)
        source_encoding: Original encoding (None for auto-detect)
        target_encoding: Target encoding (default: UTF-8)
        
    Returns:
        Number of successfully converted files
    """
    success_count = 0
    failure_count = 0
    skip_count = 0
    
    # Normalize extensions
    if extensions:
        extensions = [ext.lower() for ext in extensions]
        if not all(ext.startswith('.') for ext in extensions):
            extensions = ['.' + ext if not ext.startswith('.') else ext for ext in extensions]
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if we should process this file
            if extensions:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in extensions:
                    skip_count += 1
                    continue
                    
            if convert_file_encoding(file_path, source_encoding, target_encoding):
                success_count += 1
            else:
                failure_count += 1
    
    print(f"\nConversion summary:")
    print(f"  - Successfully converted: {success_count} files")
    print(f"  - Failed to convert: {failure_count} files")
    print(f"  - Skipped (wrong extension): {skip_count} files")
    print(f"  - Total files processed: {success_count + failure_count + skip_count}")
    
    return success_count

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python encoding_converter.py <directory_path> [--ext py,json,md] [--from utf-16] [--to utf-8]")
        return
    
    directory_path = sys.argv[1]
    
    # Parse optional arguments
    extensions = ['.py', '.json', '.md', '.log', '.txt']  # Default extensions
    source_encoding = None
    target_encoding = 'utf-8'
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--ext" and i + 1 < len(sys.argv):
            extensions = ['.' + ext if not ext.startswith('.') else ext for ext in sys.argv[i+1].split(',')]
            i += 2
        elif sys.argv[i] == "--from" and i + 1 < len(sys.argv):
            source_encoding = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--to" and i + 1 < len(sys.argv):
            target_encoding = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    print(f"Converting files in {directory_path}")
    print(f"File extensions: {extensions}")
    print(f"Source encoding: {'auto-detect' if source_encoding is None else source_encoding}")
    print(f"Target encoding: {target_encoding}")
    print("")
    
    convert_directory(
        directory_path,
        extensions=extensions,
        source_encoding=source_encoding,
        target_encoding=target_encoding
    )

if __name__ == "__main__":
    main()

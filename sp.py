import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import codecs # Using codecs for safer file reading with error handling

# --- Configuration (can be put outside the GUI class) ---
# List of directory and file names to ignore during traversal
IGNORE_LIST = [
    '.git',
    '.svn',
    '.idea',
    '.vscode',
    '__pycache__',
    'node_modules',
    'vendor',      # Common for PHP dependencies
    'target',      # Common for Maven builds
    'build',       # Common for Gradle/frontend builds
    'dist',        # Common for frontend builds
    'out',         # Common for compilation output
    'logs',
    'temp',
    '.DS_Store',   # macOS specific
    'Thumbs.db',   # Windows specific
    '*.pyc',
    '*.log',
    '*.tmp',
    '*.iml',      # IntelliJ IDEA module file
    '*.class',    # Java compiled files
    '.classpath', # Eclipse specific
    '.project',   # Eclipse specific
    '.settings',  # Eclipse specific
    'pom.xml.tag', # Backup files
    'pom.xml.bak',
    'build.gradle.lockfile',
    'package-lock.json',
    'yarn.lock',
    'composer.lock',
    'Gemfile.lock',
    'requirements.txt', # Can be ignored if you only need project code, but often useful
    'mvnw',         # Maven wrapper
    'mvnw.cmd',
    'gradlew',      # Gradle wrapper
    'gradlew.bat',
    'wrapper',      # Gradle wrapper dir
    
]

# Maximum size of a text file (in bytes) to read fully.
# Files larger than this will only have their path listed.
MAX_TEXT_FILE_SIZE = 500 * 1024 # 500 KB

# --- Core Logic Functions (from previous script, slightly adapted) ---

def is_ignored(path, ignore_list):
    """Checks if any part of the path matches an ignore pattern."""
    # Use relative path for checking against ignore list for consistency
    # Need to be careful if starting path is root or complex.
    # Simpler approach: check the base name and check if any parent dir is ignored
    path_parts = path.split(os.sep)
    for part in path_parts:
        if part in ignore_list:
            return True
        # Simple glob-like matching for filenames/extensions
        for pattern in ignore_list:
            if '*' in pattern:
                if partmatch_pattern(part, pattern):
                    return True
    return False

def partmatch_pattern(name, pattern):
    """Simple glob-like match for filename parts."""
    # Only handles patterns like '*.ext' and 'name*' for simplicity
    if pattern.startswith('*.'):
        return name.endswith(pattern[1:])
    if pattern.endswith('*'):
        return name.startswith(pattern[:-1])
    return name == pattern # Exact match handled by `in ignore_list`


def build_structure_string(directory, ignore_list, indent="", is_last=True):
    """Recursively builds the directory structure string."""
    structure = []
    # Use relative path for display if possible, or just base name
    base_name = os.path.basename(directory) if directory not in ['.', '', os.sep] else os.path.basename(os.path.abspath(directory))
    if not base_name: # Handle cases like root '/' or 'C:\'
         base_name = os.path.abspath(directory)

    # Add directory name with tree characters
    structure.append(indent + ("`-- " if is_last else "|-- ") + base_name + "/")

    # Get list of items, sorting directories first, then files, alphabetically
    try:
        items = os.listdir(directory)
        # Filter items first to correctly determine `is_last_item_filtered`
        filtered_items = [item for item in sorted(items) if not is_ignored(os.path.join(directory, item), ignore_list)]
        dirs = sorted([d for d in filtered_items if os.path.isdir(os.path.join(directory, d))])
        files = sorted([f for f in filtered_items if os.path.isfile(os.path.join(directory, f))])
        sorted_filtered_items = dirs + files

    except PermissionError:
        structure.append(indent + ("    " if is_last else "|   ") + "`-- [Permission Denied]")
        return "\n".join(structure)
    except Exception as e:
        structure.append(indent + ("    " if is_last else "|   ") + f"`-- [Error Listing: {e}]")
        return "\n".join(structure)

    num_filtered_items = len(sorted_filtered_items)
    for i, item in enumerate(sorted_filtered_items):
        path = os.path.join(directory, item)
        is_last_item_filtered = (i == num_filtered_items - 1)

        new_indent = indent + ("    " if is_last else "|   ")

        if os.path.isdir(path):
            # Recurse into subdirectory
            structure.append(build_structure_string(path, ignore_list, new_indent, is_last_item_filtered))
        else:
             # For files, just list them in the structure tree
             structure.append(new_indent + ("`-- " if is_last_item_filtered else "|-- ") + item)


    return "\n".join(structure)


def get_file_content_string(directory, ignore_list, max_size):
    """Walks through directory and reads content of non-ignored text files."""
    content_string = []
    for dirpath, dirnames, filenames in os.walk(directory, topdown=True):
        # Modify dirnames in-place to prevent os.walk from entering ignored directories
        dirnames[:] = [d for d in dirnames if not is_ignored(os.path.join(dirpath, d), ignore_list)]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Need to re-check file path itself, not just filename
            if is_ignored(filepath, ignore_list):
                continue # Skip ignored files

            content_string.append(f"\n---\nFile: {filepath}\n---\n")

            try:
                file_size = os.path.getsize(filepath)
                if file_size > max_size:
                     content_string.append(f"[File too large ({file_size/1024:.2f} KB > {max_size/1024:.2f} KB) - Content Skipped]\n")
                     continue

                # Try reading as UTF-8
                with codecs.open(filepath, 'r', encoding='utf-8', errors='strict') as f:
                     content_string.append(f.read())

            except UnicodeDecodeError:
                # Probably a binary file or different encoding
                content_string.append("[Binary or Non-UTF-8 Text File - Content Skipped]\n")
            except PermissionError:
                content_string.append("[Permission Denied - Cannot Read File]\n")
            except Exception as e:
                # Catch other potential errors during file reading
                content_string.append(f"[Error reading file: {e}]\n")

    return "".join(content_string)


def generate_project_info(project_dir, output_filename, ignore_list=None, max_size=MAX_TEXT_FILE_SIZE):
    """
    Generates project structure and file content into an output file.

    Args:
        project_dir (str): The root directory of the project.
        output_filename (str): The name of the output text file.
        ignore_list (list, optional): List of names/patterns to ignore.
                                      Defaults to the global IGNORE_LIST.
        max_size (int, optional): Max file size to read content.
                                  Defaults to MAX_TEXT_FILE_SIZE.

    Returns:
        str: A status message (success or error).
    """
    if ignore_list is None:
        ignore_list = IGNORE_LIST

    # --- Input Validation ---
    if not project_dir:
         return "Error: Please select a project directory."
    if not os.path.exists(project_dir):
        return f"Error: Directory not found at '{project_dir}'"
    if not os.path.isdir(project_dir):
        return f"Error: Path '{project_dir}' is not a directory."
    if not output_filename:
         return "Error: Please specify an output file."

    print(f"Scanning directory: {project_dir} and writing to {output_filename}") # Keep print for debugging

    # --- Build Structure String ---
    output_content = "--- Project Structure ---\n\n"

    # Start with the root directory name
    base_name = os.path.basename(os.path.abspath(project_dir))
    if not base_name: # Handle cases like root '/' or 'C:\'
         base_name = os.path.abspath(project_dir)
    output_content += base_name + "/\n"

    # Build structure of sub-items within the root
    try:
        root_items = os.listdir(project_dir)
        # Filter items first to correctly determine `is_last_item_filtered`
        filtered_root_items = [item for item in sorted(root_items) if not is_ignored(os.path.join(project_dir, item), ignore_list)]
        root_dirs = sorted([d for d in filtered_root_items if os.path.isdir(os.path.join(project_dir, d))])
        root_files = sorted([f for f in filtered_root_items if os.path.isfile(os.path.join(project_dir, f))])
        sorted_filtered_root_items = root_dirs + root_files


        num_root_items = len(sorted_filtered_root_items)
        for i, item in enumerate(sorted_filtered_root_items):
             path = os.path.join(project_dir, item)
             is_last_item = (i == num_root_items - 1)
             if os.path.isdir(path):
                  output_content += build_structure_string(path, ignore_list, "`-- " if is_last_item else "|-- ", is_last_item) + "\n"
             else:
                  # For files directly in the root, just list them in the structure
                  output_content += ("`-- " if is_last_item else "|-- ") + item + "\n"

    except PermissionError:
         output_content += "`-- [Permission Denied in root]\n"
    except Exception as e:
         output_content += f"`-- [Error Listing root: {e}]\n"


    output_content += "\n\n--- File Contents ---\n"

    # --- Get File Contents String ---
    # Use a try-except block around the content generation as well
    try:
        output_content += get_file_content_string(project_dir, ignore_list, max_size)
    except Exception as e:
        output_content += f"\n--- Error generating file contents: {e} ---\n"


    # --- Write to Output File ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(output_content)
        return f"Successfully generated '{output_filename}'"
    except IOError as e:
        return f"Error writing to output file '{output_filename}': {e}"
    except Exception as e:
        return f"An unexpected error occurred during writing: {e}"


# --- Tkinter GUI Application ---

class ProjectInfoGeneratorApp:
    def __init__(self, root):
        self.root = root
        root.title("Project Info Generator")
        root.geometry("600x300") # Set a default window size

        # Use ttk for themed widgets
        self.style = ttk.Style()
        self.style.theme_use('clam') # Or 'default', 'alt', 'classic'

        # --- Input Directory ---
        self.frame_input = ttk.Frame(root, padding="10")
        self.frame_input.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame_input.columnconfigure(1, weight=1) # Allow entry field to expand

        self.label_input = ttk.Label(self.frame_input, text="Project Directory:")
        self.label_input.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.entry_input = ttk.Entry(self.frame_input, width=50)
        self.entry_input.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.button_input = ttk.Button(self.frame_input, text="Browse...", command=self.browse_input_directory)
        self.button_input.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

        # --- Output File ---
        self.frame_output = ttk.Frame(root, padding="10")
        self.frame_output.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame_output.columnconfigure(1, weight=1) # Allow entry field to expand

        self.label_output = ttk.Label(self.frame_output, text="Output File:")
        self.label_output.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.entry_output = ttk.Entry(self.frame_output, width=50)
        self.entry_output.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.button_output = ttk.Button(self.frame_output, text="Save As...", command=self.browse_output_file)
        self.button_output.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

        # --- Generate Button ---
        self.frame_generate = ttk.Frame(root, padding="10")
        self.frame_generate.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame_generate.columnconfigure(0, weight=1) # Center the button if frame expands

        self.button_generate = ttk.Button(self.frame_generate, text="Generate Project Info", command=self.start_processing)
        self.button_generate.grid(row=0, column=0, padx=5, pady=10) # Use padx/pady for button spacing

        # --- Status Label ---
        self.frame_status = ttk.Frame(root, padding="10")
        self.frame_status.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame_status.columnconfigure(0, weight=1)

        self.label_status = ttk.Label(self.frame_status, text="Ready", anchor="w")
        self.label_status.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # --- Configure Grid Weights for Resizing ---
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0) # Input frame doesn't grow vertically
        root.rowconfigure(1, weight=0) # Output frame doesn't grow vertically
        root.rowconfigure(2, weight=0) # Generate frame doesn't grow vertically
        root.rowconfigure(3, weight=1) # Status frame can grow vertically

    def browse_input_directory(self):
        """Opens a dialog to select the project directory."""
        directory_path = filedialog.askdirectory(title="Select Project Directory")
        if directory_path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, directory_path)

            # Suggest an output filename based on the selected directory
            base_name = os.path.basename(directory_path)
            suggested_output_name = f"{base_name}_project_info.txt" if base_name else "project_info.txt"
            # Only suggest if output field is empty
            if not self.entry_output.get():
                 self.entry_output.delete(0, tk.END)
                 self.entry_output.insert(0, suggested_output_name)


    def browse_output_file(self):
        """Opens a dialog to save the output file."""
        initial_dir = os.path.dirname(self.entry_input.get()) if os.path.isdir(self.entry_input.get()) else os.getcwd()
        initial_file = self.entry_output.get() if self.entry_output.get() else "project_info.txt"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=initial_file,
            title="Save Project Info As"
        )
        if file_path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, file_path)

    def start_processing(self):
        """Initiates the project scanning and file generation."""
        project_dir = self.entry_input.get()
        output_file = self.entry_output.get()

        # Basic validation
        if not project_dir:
            messagebox.showwarning("Input Error", "Please select the project directory.")
            return
        if not output_file:
            messagebox.showwarning("Input Error", "Please specify the output file.")
            return

        # Update status and disable button
        self.label_status.config(text="Processing...")
        self.button_generate.config(state=tk.DISABLED)
        self.root.update_idletasks() # Update GUI immediately

        # Perform the generation
        try:
            # Call the core logic function
            status_message = generate_project_info(project_dir, output_file, IGNORE_LIST, MAX_TEXT_FILE_SIZE)
            self.label_status.config(text=status_message)
            if "Successfully generated" in status_message:
                 messagebox.showinfo("Success", status_message)
            else:
                 messagebox.showerror("Error", status_message) # Show error in a popup

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            self.label_status.config(text=f"Error: {e}")
            messagebox.showerror("Unexpected Error", error_message)

        finally:
            # Re-enable button
            self.button_generate.config(state=tk.NORMAL)


# --- Main execution block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectInfoGeneratorApp(root)
    root.mainloop()
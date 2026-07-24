class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    PURPLE = '\033[95m'
    YELLOW = '\033[93m'
    DARKCYAN = '\033[36m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'     
    BOLD = '\033[1m'

def log_info(msg: str, color: str = Colors.OKBLUE):
    """Logs an info message. Defaults to blue, but accepts custom color codes."""
    print(f"{color}[INFO] {msg}{Colors.ENDC}")

# Optional: Update other functions similarly if needed
def log_header(msg: str):
    print(f"{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")

def log_success(msg: str):
    print(f"{Colors.OKGREEN}[SUCCESS] {msg}{Colors.ENDC}")

def log_warning(msg: str):
    print(f"{Colors.WARNING}[WARNING] {msg}{Colors.ENDC}")

def log_error(msg: str):
    print(f"{Colors.FAIL}[ERROR] {msg}{Colors.ENDC}")
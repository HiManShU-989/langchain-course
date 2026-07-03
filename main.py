import re
import inspect
from dotenv import load_dotenv
load_dotenv() 
import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.5:latest"

# Langcahin tool decorators allow us to easily create tools that can be called by the agent.
# But using raw functions we have to write provider specific code to make them work with the agent.
# The traceable decorator from langsmith allows us to easily create tools that can be called by the agent 
# without having to write provider specific code. It also allows us to track the execution of the tools in the langsmith dashboard.


@traceable(run_type = "tool")
def get_product_price(product:str) -> float:
    """Look up the price of a product in the catalog.
    
    args:
        product: The name of the product to look up.
    returns:
        The price of the product.
    """
    print(f"    >> Executing the get_product_price(product = '{product}')")
    prices = {
        "laptop":1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }
    return prices.get(product, 0.0)

@traceable(run_type = "tool")
def apply_discount(price:float, discount_tier:str) -> float:
    """Apply a discount to a price based on the discount tier and return the final price.
    
    args:
        price: The original price of the product.
        discount_tier: Available tiers ("bronze" "silver", "gold").
    returns:
        The discounted price of the product.
    """
    price = float(price)
    print(f"    >> Executing the apply_discount(price = {price}, discount_tier = '{discount_tier}')")
    discounts = {
        "silver": 12,
        "gold": 23,
        "bronze": 5
    }
    discount_rate = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount_rate / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ attribute is used to access the original function wrapped by the decorator.
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature}- {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# Without langchain, we have to manually trace the LLM calls.
@traceable(name="Ollama Chat", run_type = "llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model = model, messages = messages, options = options)

# _______AGENT LOOP _________
@traceable(name = "Ollama agent loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("="*60)
    prompt = react_prompt.format(question=question)
    scratchpad = ""
    
    for iteration in range(1,MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role":"user", "content":full_prompt}],
            options = {"stop":["\nObservation"], "temperature": 0.0}
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")
        
        print(f"[Parsing] Looking for final answer in LLM output")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"[Parsed] Final Answer found: {final_answer}")
            print("\n"+"="*60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        # Parse tool calls from raw text using regex- fragile if LLM doesn't follow format.
        print(f"[Parsing] Looking for tool calls in LLM output")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
       
        if(not action_match or not action_input_match):
           print(f"[Parsing] No tool calls found in LLM output. Stopping agent.")
           break
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        print(f"  [Tool selected]: {tool_name} with args: {tool_input_raw}")
        # Split comma separated args strip key= prefix if LLM outputs key=value format. Very fragile if LLM doesn't follow format.
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]
        
        print(f"[Tool Executing] {tool_name}({args})...")
        
        if tool_name not in tools:
            observation = f"Error:Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
            
        print(f" Tool Result: {observation}")
        
        # History is one growing string re-sent every iteration(replaces messages.append)
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
       
    print("ERROR: Max iterations reached without a final answer.")
    return None
        
        


if __name__ == "__main__":
    print("Hello langchain Agent(.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)
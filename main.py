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

tools_for_llm = [
    {
        "type": "function",
        "function":{
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up e.g. 'laptop', 'headphones', 'keyboard'."
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
       "type": "function",
       "function": {
           "name": "apply_discount",
           "description": "Apply a discount to a price based on the discount tier and return the final price.",
           "parameters": {
               "type": "object",
               "properties": {
                   "price": {
                       "type": "number",
                       "description": "The original price of the product."
                   },
                   "discount_tier": {
                       "type": "string",
                       "description": "The discount tier to apply. Available tiers are 'bronze', 'silver', and 'gold'."
                   }
               },
               "required": ["price", "discount_tier"]
           }
       } 
    }
] 
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
    print(f"    >> Executing the apply_discount(price = {price}, discount_tier = '{discount_tier}')")
    discounts = {
        "silver": 12,
        "gold": 23,
        "bronze": 5
    }
    discount_rate = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount_rate / 100), 2)


# Without langchain, we have to manually trace the LLM calls.
@traceable(name="Ollama Chat", run_type = "llm")
def ollama_chat_traced(messages):
    return ollama.chat(model = MODEL, messages = messages, tools = tools_for_llm)



# _______AGENT LOOP _________
@traceable(name = "Ollama agent loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    
    # On using langchain, we don't have to worry about the prompt formatting or how to pass tool results back to the LLM.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant."
                      "You have access to a product catalog tool"
                      "and a discount tool.\n\n"
                      "STRICT RULES- you must follow these exactly:\n"
                      "1. Never guess or assume any product price."
                      "You must call get_product_price first to get the real price.\n"
                      "2. Only call apply_discount AFTER you have received"
                      "a price from get_product_price. Pass the exact price"
                      "returned by get_product_price - do NOT pass a made-up number.\n"
                      "3. NEVER calculate discount yourself using math."
                      "Always use the apply_discount tool.\n"
                      "4. If the user does not specify a discount tier,"
                      "ask them which tier to use - do NOT assume one."
        },
        {
            "role":"user",
            "content":question
        }
    ]
    
    for iteration in range(1,MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        # Ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        # if no tool calls means we have the final answer, break the loop and return the answer
        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content
        
        #Process only first tool call- force one tool per iteration for simplicity
        tool_call = tool_calls[0]
        # Difference in attribute access method.
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f" [Tool Selected]: {tool_name} with args: {tool_args} ")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in available tools.")
        
        # Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)
        print(f" [Tool Result]: {observation} \n")
        
        messages.append(ai_message)
        messages.append({
            "role":"tool",
            "content":str(observation)
        })
            
    print("ERROR: Max iterations reached without a final answer.")
    return None
        
        


if __name__ == "__main__":
    print("Hello langchain Agent(.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)
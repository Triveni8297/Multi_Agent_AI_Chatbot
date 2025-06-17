import os
import json
import uuid  # For generating unique thread_id
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import f1_faiss_search, f2_pinecone_search, f3_google_search


# === Load Environment Variables ===#
load_dotenv()


# === System Prompt for Tool Selection (Updated) ===
system_prompt = (
    "You are a smart and polite assistant that selects the appropriate tool based on the query type:\n\n"
    "✅ Use **f1_faiss_search** when the query is related to software training centers (e.g., Data Science, Deep Learning, Full Stack Development, etc.), including details like fees, curriculum, services, contact information, address, and location.\n"
    "✅ Use **f2_pinecone_search** when the query is about Generative AI, its concepts, use cases, techniques, or related topics.\n"
    "✅ Use **f3_google_search** only when the query asks for **latest sports news, live scores, or current sporting events**.\n\n"
    "❗ If the question is unrelated to software training, generative AI, or current sports news, respond with: "
    "'I'm sorry, I can currently assist only with queries related to software training courses, generative AI, or the latest sports events. Please ask something relevant to these topics.'\n\n"
    "📌 Always match the query to the most appropriate tool and invoke only one.\n"
    "📌 If the query is about general definitions already available in stored documents (e.g., what is GAN?), prefer **f2_pinecone_search**.\n"
    "📌 Never use **f3_google_search** for anything other than current sports updates.\n\n"
    "🎉 For greetings (e.g., 'hello', 'hi'), introductions (e.g., 'who are you?', 'what can you do?'), and farewells (e.g., 'bye', 'thank you'), respond in a warm, friendly, and polite manner, without invoking any tools.\n"
    "\n"
    "📝 All responses must be returned in valid JSON format as follows: {\"response\": <your_reply>}\n"
)


# === Instantiate tools and wrap callbacks to log usage ===#
tool_instances = [f1_faiss_search, f2_pinecone_search, f3_google_search]

# === LLM and Agent Setup ===#
model = init_chat_model("gpt-4o-mini", model_provider="openai")
memory = MemorySaver()

# === Generate a dynamic unique thread_id ===#
unique_thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": unique_thread_id}}
print(f"🧵 Thread ID: {unique_thread_id}")

# === Function to Run the Agent and Log Tool Use ===#
def run_agent(query: str):
    try:
        agent_executor = create_react_agent(
            prompt=system_prompt,
            model=model,
            tools=tool_instances,
            checkpointer=memory
        )
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        final_response = response["messages"][-1].content
        # Ensure the response is parsed as JSON and handle string output
        final_response_json = json.loads(final_response)
        print("**************************************************************")
        print(f"🤖 Running agent with query: {query}")
        print("✅ Final response:", final_response_json["response"])
        print("**************************************************************")
        return final_response_json["response"]
    except Exception as e:
        print(f"❌ Error while running agent: {e}")
        return "Sorry, something went wrong while processing your query."

# === Run When File is Executed ===#
if __name__ == "__main__":
    sample_query = "Who won the recent ipl match?"
    run_agent(sample_query)


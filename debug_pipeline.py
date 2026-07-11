from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

def main():
    topic = "explainable AI in medical imaging"

    print("=== Running Search Agent ===")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [{"role": "user", "content": f"Find recent academic papers about: {topic}"}]
    })
    research_text = search_result["messages"][-1].content
    print(research_text[:1000])  # preview

    print("\n=== Running Writer Chain ===")
    report = writer_chain.invoke({"topic": topic, "research": research_text})
    print(report[:1000])  # preview

    print("\n=== Running Critic Chain ===")
    critique = critic_chain.invoke({"report": report})
    print(critique)

if __name__ == "__main__":
    main()
from main import workflow


if __name__ == "__main__":
    while True:
        human = input("\n\nHuman- ")
        if human == "exit":
            break
        print()
        initial_state = {
            "user_query": human,
            "k": 3,
            "max_retry_for_revise_answer": 3,
            "max_retry_for_rewrite_query": 2,
        }

        print("RUNNING WORKFLOW")
        for chunk in workflow.stream(
            initial_state, {"configurable": {"thread_id": "2"}}, stream_mode="updates"
        ):
            print(f"{list(chunk.keys())[0]} is completed")

        response = workflow.get_state(config={"configurable": {"thread_id": "2"}})
        print("\nAI- ", response.values["generated_response"])
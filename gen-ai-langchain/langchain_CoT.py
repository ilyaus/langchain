from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


DEBUG = False


def debug_response(x):
    if DEBUG:
        print(f"DEBUG: {x}")
    return x


def main():
    stance_template_1 = """
        Stance classification is the task of determining the expressed or 
        implied opinion, or stance, of the reply towards the comment.
        comment: {comment}
        reply: {reply}
        explanation:
    """
    stance_template_2 = """
        Therefore, based on your explanation, {stance_reason}, what is the final stance? 
        Respond with a single word: "agree", "disagree", or "neutral". 
        Only return the stance as a single word, and no other text.
        comment: {comment}
        reply: {reply}
        stance:
    """

    stance_prompt_1 = PromptTemplate(
        template=stance_template_1, input_variables=["comment", "reply"]
    )

    stance_prompt_2 = PromptTemplate(
        template=stance_template_2, input_variables=["comment", "reply", "stance_reason"]
    )
    
    # -- Model ---
    hf_raw = pipeline(
        "text2text-generation",
        model="declare-lab/flan-alpaca-gpt4-xl",
        device=-1,  # 0: GPU; -1: CPU
        max_new_tokens=500,
    )

    # hf_raw = pipeline(
    #     "text-generation",
    #     model="tiiuae/Falcon3-3B-Instruct",
    #     device=-1,
    #     max_new_tokens=500,
    #     return_full_text=False
    # )

    llm = HuggingFacePipeline(pipeline=hf_raw)
    parser = StrOutputParser()

    chain_1 = stance_prompt_1 | llm | debug_response | parser
    chain_2 = stance_prompt_2 | llm | debug_response | parser
    cot_chain = {"stance_reason": chain_1, 
                 "comment": RunnablePassthrough(), 
                 "reply": RunnablePassthrough()} | chain_2

    # --- Examples ---
    comment = "The new Dune movie does not really capture the vision laid out by Frank Herbert. It feels like they tried to import too many visual effects that take away from the philosophy of the work."
    replies = [
        "The newer ones fail to live up to the sophistry of the older movies from the 70's.",
        "Frank Herbert wrote a lot of books.",
        "I think the new Dune movie better captures the spirit, if not the content, of Frank Herbert's philosophy.",
        "The quick red fox jumped over the lazy brown dog.",
        "Yeah, this new movie is a real masterpiece, lol!!",
    ]

    print("Model Output:")
    for reply in replies:
        result = cot_chain.invoke({"comment": comment, "reply": reply})
        print(reply)
        print(result)


if __name__ == "__main__":
    main()
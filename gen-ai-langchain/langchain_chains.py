from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


def debug_response(x):
    print(x)

    return x


def main():
    stance_template = """
    Please classify the stance, or opinion, of the following reply to 
    the comment. Note that we want the stance of the reply **to the comment**, 
    not the stance of the reply to the topic of the comment.

    Only output exactly one word: "agree", "disagree", or "neutral".
    
    Comment: {comment}
    Reply: {reply}
    Stance:
    """

    stance_prompt = PromptTemplate(
        template=stance_template, input_variables=["comment", "reply"]
    )

    # -- Model ---
    # hf_raw = pipeline(
    #     "text2text-generation",
    #     model="declare-lab/flan-alpaca-gpt4-xl",
    #     device=-1,  # 0: GPU; -1: CPU
    #     max_new_tokens=500,
    # )

    hf_raw = pipeline(
        "text-generation",
        model="tiiuae/Falcon3-3B-Instruct",
        device=-1,
        max_new_tokens=500,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=hf_raw)

    # --- Output Parser ---
    parser = StrOutputParser()

    chain = stance_prompt | llm | debug_response | parser

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
        result = chain.invoke({"comment": comment, "reply": reply})    
        print(result)


if __name__ == "__main__":
    main()

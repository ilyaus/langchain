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
    # -- Model ---
    # hf_raw = pipeline(
    #     "text2text-generation",
    #     model="declare-lab/flan-alpaca-gpt4-xl",
    #     device=-1,    # 0: GPU; -1: CPU
    #     max_new_tokens=5000,
    # )

    hf_raw = pipeline(
        "text-generation",
        model="tiiuae/Falcon3-3B-Instruct",
        device=-1,  # 0: GPU; -1: CPU
        max_new_tokens=5000,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=hf_raw)
    parser = StrOutputParser()

    gen_hypothesis = """
        Consider the following comment and reply:
        comment: {comment}
        reply: {reply}

        Generate three different hypotheses for the stance of the reply towards the comment.
        For each hypothesis, explain why the reply might:
        1. Agree
        2. Disagree
        3. Be Neutral

        Output each hypothesis clearly labeled (e.g., "Hypothesis 1: ...") with a newline 
        between each hypothesis.
    """
    gen_hypothesis_prompt = PromptTemplate(
        template=gen_hypothesis, input_variables=["comment", "reply"]
    )
    gen_hypothesis_chain = gen_hypothesis_prompt | llm

    eval_hypothesis = """
        Consider the following comment and reply:
        comment: {comment}
        reply: {reply}

        Given the following hypotheses and explanations for the stance of the reply towards the comment:
        {hypotheses}

        Evaluate each hypothesis based on its logical consistency and support from the reply.
        Assign a numerical score from 1 to 5 for each hypothesis, where 5 is highly consistent and 1 is 
        inconsistent. Only reply with the score and reason for that score for each hypothesis 
        (e.g., Hypothesis 1: [score], reason: ...) and no other text.
    """
    eval_hypothesis_prompt = PromptTemplate(
        template=eval_hypothesis, input_variables=["comment", "reply", "hypotheses"]
    )
    eval_hypothesis_chain = eval_hypothesis_prompt | llm

    make_decision = """
        Consider the following comment and reply:
        comment: {comment}
        reply: {reply}

        Based on the evaluations of different hypotheses for the stance of the reply towards the comment:
        {hypotheses}

        {evaluations}

        Select the hypothesis with the highest score. Output the final stance as 
        "agree", "disagree", or "neutral" based on the chosen hypothesis. 
        Only output the label as single word and do not generate any other text after the label.
        
        label:
        """
    make_decision_prompt = PromptTemplate(
        template=make_decision,
        input_variables=["comment", "reply", "hypotheses", "evaluations"],
    )
    make_decision_chain = make_decision_prompt | llm

    tot_chain = {
        "hypotheses": gen_hypothesis_chain,
        "comment": RunnablePassthrough(),
        "reply": RunnablePassthrough(),
    } | RunnablePassthrough.assign(evaluations = eval_hypothesis_chain) | make_decision_chain

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
        result = tot_chain.invoke({"comment": comment, "reply": reply})
        print(reply)
        print(result)


if __name__ == "__main__":
    main()

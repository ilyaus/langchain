from langchain_core.prompts import PromptTemplate
from transformers import pipeline

def main():
    stance_template = """
    Please classify the stance, or opinion, of the following reply to 
    the comment.  Note that we want the stance of teh reply to the comment, 
    and not the stance fo teh reply to topic of the comment.  
    Only give the stance as "agree", "disagree", or "neutral" adn output 
    no other words after outputting the label.  
    Comment: {comment} 
    Reply: {reply} 
    Stance: """

    stance_prompt = PromptTemplate(template=stance_template, input_variables=["comment", "reply"])

    comment = "I think climate change is the most pressing issue of our time."
    reply = "I completely not agree, we may not need to take immediate action to address it."
    # reply = "We may need more love."


    prompt_value = stance_prompt.format(comment=comment, reply=reply)
    print(prompt_value)

    hf_pipeline = pipeline(
        "text2text-generation",
        model="declare-lab/flan-alpaca-gpt4-xl",
        device=-1, # 0: GPU; -1: CPU
        max_new_tokens=500
    )

    # hf_pipeline = pipeline(
    #     "text-generation",bb
    #     model="tiiuae/Falcon3-3B-Instruct",
    #     device=-1,
    #     max_new_tokens=500,
    #     return_full_text=False
    # )

    # The HuggingFacePipeline wrapper may not implement __call__ in this
    # environment, so call the underlying Hugging Face `pipeline` directly.
    model_output = hf_pipeline(prompt_value)

    print(f"Generated text: {model_output}")
    print()

    # Normalize common pipeline return shapes and print the generated text
    if isinstance(model_output, list) and len(model_output) > 0:
        first = model_output[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("text") or str(first)
        else:
            text = str(first)
    else:
        text = str(model_output)

    print(text)


if __name__ == "__main__":
    main()

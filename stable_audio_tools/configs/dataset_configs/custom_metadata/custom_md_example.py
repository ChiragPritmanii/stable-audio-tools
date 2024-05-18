def get_custom_metadata(info, audio):
    concat = info['prompt'] + f" keywords: {", ".join(info['keywords'])}"
    sample_info = {"concat_prompt": concat}
    return sample_info

    # # Use relative path as the prompt
    # return {"prompt": info["relpath"]}
# can get rid of this 
def transform(sample_json, audio):
    concat = sample_json['prompt'] + f" keywords: {", ".join(sample_json['keywords'])}"
    sample_info = {"concat_prompt": concat}
    return sample_info

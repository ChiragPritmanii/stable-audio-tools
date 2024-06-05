import random 

def get_custom_metadata(info, audio):
    tags = [info["artist"]["tags"], info["track"]["tags"], info["track"]["genres"]]
    #avoiding album text, because its too large and often less descriptive
    text = [info["album"]["title"], info["track"]["title"], info["track"]["info"], info["track"]["year"]]
    filtered_text = []
    for i in text:
        if i!="":
            filtered_text.append(i)
    
    filtered_tags = []
    for i in tags:
        if len(i)>0:
            filtered_tags+=i
    
    complete_info = filtered_tags+filtered_text
    random.shuffle(complete_info) #shuffle the list inplace 
    prompt = ", ".join(complete_info)          
    return {"prompt": prompt}

    # # Use relative path as the prompt
    # return {"prompt": info["relpath"]}
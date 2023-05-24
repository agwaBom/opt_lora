
possible_substring_list = [
    # du-en
    "I'm not sure if this is a good idea.",
    "I'm not sure if it's a good idea to have a \"no-no\" list.",
    "I think it's a good idea to have a \"no-no\" list",
    "but I think it's a bad idea to have a \"no-no\" list",
    "but I think it should be a list of things that are allowed",
    "but I think it should be a list of things that are not allowed",
    "not a list of things that are allowed",
    "not a list of things that are not allowed",
    "I'm not sure if you're trying to be funny or not, but I think it's a bad idea",
    "that is so long that it's impossible to read",
    "I think it's better to have a list that is short and to the point",

    # en-du
    "I think it's a good idea to have a list of things that are allowed",
    "but I think it's a bad idea to have a list of things that are not allowed",
    "I think it's better to have a list of things that are not allowed, but",
    "I think it's better to have a list of things that are not allowed, and then a list of things that are allowed.",
    "I think it's a bad idea to have a \"no-no\" list.",
    "I'm not sure if this is a joke or not, but I'm pretty sure that's a real thing.",
    "I think it's a good idea, but I think it's a bad idea.  I think it's a bad idea because it's a bad idea.  I think it's a bad idea because it's a bad idea.  I think it's a bad idea because it's a bad idea.  I think it's a bad idea because it's a bad idea.  I think it's a bad idea because it's a bad idea.",
    "I think it's a bad idea because it's a bad idea.",
    "I think it's a good idea, but I think it's a bad idea",

    "I'm not sure if this is the right place to ask,",
    "I'm not sure if this is a good thing or a bad thing."

]

possible_redundant_list = [
    " . ",
    " , ",
    " ,",
    " ."
]

if __name__ == "__main__":
    path = "./facebook_opt-iml-max-1.3b_en_du_pred_list_fixed.txt"
    
    file = open(path, mode='r').read()
    for s in possible_substring_list:
        file = file.replace(s, "")

    for s in possible_redundant_list:
        file = file.replace(s, "")
    for s in possible_redundant_list:
        file = file.replace(s, "")


    with open("./facebook_opt-iml-max-1.3b_en_du_pred_list_fixed_output.txt", mode='w') as out:
        out.write(file)
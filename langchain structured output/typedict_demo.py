from typing import TypedDict

class hello(TypedDict):
    name:str
    age:int

new_person: hello = {"name":"ayan","age":56}

print(new_person)
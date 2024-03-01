from response_generator.Response_Generator import Response_Generator
from vector_store.Vector_Store import Vector_Store
from ui.Web_Interface import Web_Interface

# You can test the different elements independently here:

# Step 1
# vec = Vector_Store()
# query = "What name does my cat have?"
# docs = vec.get_retriever().get_relevant_documents(query)
# print(docs[0])

# Step 2 (might throw some errors)
# r_gen = Response_Generator()
# res = r_gen.generate_response("What name does my cat have?")
# print(res)

# Step 3
ui = Web_Interface()
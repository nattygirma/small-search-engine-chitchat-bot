import re
from tqdm import tqdm
from preprocessor import Preprocessor
from indexer import Indexer
from multiTopicIndexer import MultiTopicIndexer
from collections import OrderedDict
from linkedList import LinkedList
import inspect as inspector
import json
import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI
import json
import random
from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()
        self.multi_topic_indexer = MultiTopicIndexer()

    def run_indexer_from_specific_field(self, json_file):

        topic_name = os.path.splitext(os.path.basename(json_file))[0]

        with open(json_file, 'r', encoding='utf-8') as file:
            # Load the JSON content
            corpus = json.load(file)

    # Ensure the topic exists in the JSON file
        if topic_name not in corpus:
            raise ValueError(f"Field '{topic_name}' not found in the JSON file '{json_file}'.")

        # Process documents under the specific topic
        documents = corpus[topic_name]
        
        for doc_id, document in enumerate(documents, start=1):
            # Tokenize the document
            # print(document)
            tokenized_document = self.preprocessor.tokenizer(document['summary'])
        
            # Add the document to the appropriate topic index
            self.multi_topic_indexer.add_document(topic_name, doc_id, tokenized_document)

        # Process the topic index
        self.multi_topic_indexer.process_index(topic_name)
        print(f"Indexing complete for topic '{topic_name}' K")


    def run_indexer(self, corpus):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
            Already implemented, but you can modify the orchestration, as you seem fit."""
        with open(corpus, 'r', encoding='utf-8') as file:
        # print(file)
            for line in file:
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                # print("tokenized documents",tokenized_document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()
        print("Indexing Complete")
        # for term, posting_list in self.indexer.inverted_index.items():
        # # posting_list.build_skip_pointers()
        #      postings = posting_list.traverse()
        #      docFreq= posting_list.docFreq
        #     #  print(f"Term: `{term}`, is in {docFreq} Documents, Postings: {postings}")


    def run_queries(self, query, topic):
        """ DO NOT CHANGE THE output_dict definition"""
        output_dict = {
                #     'postingsList': {},
                #     'postingsListSkip': {},
                #    'daatAnd': {},
                #    'daatAndSkip': {},
                #    'daatAndTfIdf': {},
                   'daatAndSkipTfIdf': {},
                   }
        

        if(True):
            # 1. Pre-process & tokenize the query
            input_term_arr = self.preprocessor.preprocess_query(query)
            print("0")

            # Initialize lists for storing postings and skip postings
            daat_and_result = None
            daat_and_skip_result = None
            daat_and_skip_result_l = None

            daat_and_comparisons = 0
            daat_and_skip_comparisons = 0

            for term in input_term_arr:
                # 2. Retrieve postings list and skip postings list for each term
                # postings_list = self.get_postings_list(term,topic)
                skip_postings_list = self.get_skip_postings_list(term, topic)
                skip_postings_linkedList = self.get_skip_postings_linkedList(term, topic)

                # if postings_list is None:
                #     postings_list = []  # Handle missing term by setting an empty list
                if skip_postings_list is None:
                    skip_postings_list = []
                if skip_postings_linkedList is None:
                    skip_postings_linkedList = LinkedList()
            # Store postings list and skip postings list in the output dict
                # output_dict['postingsList'][term] = [doc_id for doc_id, _ in postings_list]
                # output_dict['postingsListSkip'][term] = [doc_id for doc_id, _ in skip_postings_list]
            
                # 3. For DAAT AND operation, intersect postings list and skip postings list
                # if daat_and_result is None:
                #     daat_and_result = postings_list
                # else:
                #     daat_and_result, comparisons = self.daat_and(daat_and_result, postings_list)
                #     daat_and_comparisons += comparisons

                if daat_and_skip_result_l is None:
                    daat_and_skip_result_l = skip_postings_linkedList
                else:
                   daat_and_skip_result_l, comparisons1 = self.daat_and_skip(daat_and_skip_result_l, skip_postings_linkedList)
                   daat_and_skip_comparisons += comparisons1
                daat_and_skip_result = self.linkedlist_to_array(daat_and_skip_result_l)
          
            # 4. Sort DAAT AND results by TF-IDF (if needed)
            print("1")
            # daat_and_tfidf_result = self.sort_by_tfidf(daat_and_result)
            daat_and_skip_tfidf_result = self.sort_by_tfidf(daat_and_skip_result)
            # daat_and_skip_tfidf_result = self.sort_by_tfidf(daat_and_result)

            # Calculate number of docs in results
            # num_docs_daat = len(daat_and_result)
            # print("2")
            # # num_docs_daat_skip = len(daat_and_result)
            # num_docs_daat_skip = len(daat_and_skip_result)
            # num_docs_daat_tfidf = len(daat_and_tfidf_result)
            # num_docs_daat_skip_tfidf = len(daat_and_skip_tfidf_result)
            # print("3")

            # Store the DAAT AND results and their tf-idf sorted versions with the required format
            # output_dict['daatAnd'][query] = {
            #     'num_comparisons': daat_and_comparisons,
            #     'num_docs': num_docs_daat,
            #     'results': [doc_id for doc_id, _ in daat_and_result]
            # }

            # output_dict['daatAndSkip'][query] = {
            #     'num_comparisons': daat_and_skip_comparisons,
            #     'num_docs': num_docs_daat_skip,
            #     'results': [doc_id for doc_id, _ in daat_and_skip_result]
            # }

            # output_dict['daatAndTfIdf'][query] = {
            #     'num_comparisons': daat_and_comparisons,  # Same as DAAT AND
            #     'num_docs': num_docs_daat_tfidf,
            #     'results': [doc_id for doc_id, _ in daat_and_tfidf_result]
            # }

            output_dict['daatAndSkipTfIdf'] = {
                'num_comparisons': daat_and_skip_comparisons,  # Same as DAAT AND with Skip
                # 'num_docs': num_docs_daat_skip_tfidf,
                'results': [doc_id for doc_id, _ in daat_and_skip_tfidf_result],
                'everything':daat_and_skip_tfidf_result
            }

        # output_dict['postingsList'] = dict(sorted(output_dict['postingsList'].items()))
        
        # output_dict['postingsListSkip'] = dict(sorted(output_dict['postingsListSkip'].items()))
        

        return output_dict
    
    def get_postings_list(self, term, topic):
        """Retrieve the postings list for a term within the specified topic."""
        if topic in self.multi_topic_indexer.topic_indexes:
            topic_index = self.multi_topic_indexer.topic_indexes[topic]
            return topic_index.inverted_index.get(term, None).traverse() if term in topic_index.inverted_index else None
        return None

    def get_skip_postings_linkedList(self, term, topic):
        """Retrieve the LinkedList with skip pointers for a term within the specified topic."""
        if topic in self.multi_topic_indexer.topic_indexes:
            topic_index = self.multi_topic_indexer.topic_indexes[topic]
            return topic_index.inverted_index.get(term, None) if term in topic_index.inverted_index else None
        return None

    def get_skip_postings_list(self, term, topic):
        """Retrieve the postings list with skip pointers for a term in array format for the specified topic."""
        if topic in self.multi_topic_indexer.topic_indexes:
            topic_index = self.multi_topic_indexer.topic_indexes[topic]
            return topic_index.inverted_index.get(term, None).traverse_with_skips() if term in topic_index.inverted_index else None
        return None
    def linkedlist_to_array(self, linked_list: LinkedList):
        """Convert a LinkedList to an array of tuples (doc_id, term_freq)."""
        array_result = []
        current_node = linked_list.head
        while current_node is not None:
           # Append (doc_id, term_freq) to the result array
           array_result.append((current_node.doc_id, current_node.tf_idf))
           current_node = current_node.next

        return array_result
    def daat_and(self, postings1, postings2):
        """Perform DAAT AND operation on two postings lists."""
        result = []
        comparisons = 0
        i, j = 0, 0

        while i < len(postings1) and j < len(postings2):
             comparisons += 1  # Increment comparisons for each comparison
             doc1 = postings1[i][0]
             doc2 = postings2[j][0]
             
             if doc1 == doc2:
                
                result.append(postings1[i])  # Add doc_id to result if they match
                i += 1
                j += 1
             elif doc1 < doc2:
                 i += 1  # Move p1 to the next node
             else:
                j += 1  # Move p2 to the next node

        return result, comparisons


    def daat_and_skip(self, postings1: LinkedList, postings2: LinkedList):
        """Perform DAAT AND operation with skip pointers on LinkedList objects."""
        result = LinkedList()  # Use a LinkedList to store the result
        comparisons = 0
 
        p1 = postings1.head
        p2 = postings2.head
 
        while p1 is not None and p2 is not None:
            comparisons += 1
            doc1 = p1.doc_id
            doc2 = p2.doc_id

            # print(doc1 , " and ", doc2)

            if doc1 == doc2:
             
                # If both doc IDs match, add to result and move both pointers
                result.insert(p1.doc_id, p1.term_freq, p1.tf, p1.tf_idf)
                p1 = p1.next
                p2 = p2.next
            elif doc1 < doc2:
                # Check if we can skip in postings1
                if p1.skip and p1.skip.doc_id <= doc2:
                    # Move using skip pointers in postings1 as long as skip target is <= doc2
                    while p1.skip and p1.skip.doc_id <= doc2:
              
                        p1 = p1.skip  # Move to the skip target
                else:
                    # Move to the next posting if no skip is useful
                    p1 = p1.next
            else:
                # Check if we can skip in postings2
                if p2.skip and p2.skip.doc_id <= doc1:
                    # Move using skip pointers in postings2 as long as skip target is <= doc1
                    while p2.skip and p2.skip.doc_id <= doc1:
                     
                        p2 = p2.skip  # Move to the skip target
                else:
                    # Move to the next posting if no skip is useful
                    p2 = p2.next

        return result, comparisons
    def sort_by_tfidf(self, postings_list):
        """Sort the postings list by TF-IDF score, return empty list if None."""
        if postings_list is None:
            return []  # If postings_list is None, return an empty list
        return sorted(postings_list, key=lambda x: x[1], reverse=True)  # Assumin
    # 
    
    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """

        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

def saveOutput(output, topic):
    # Load the food.json file
    with open(f'{topic}.json', 'r', encoding='utf-8') as food_file:
        food_data = json.load(food_file)

    # Extract the "Food" list from food_data
    food_list = food_data[topic]

   # Get the results from daatAndSkipTfIdf for the topic 'British cuisine'
    doc_ids = output['daatAndSkipTfIdf']['results'][:10]
    everything = output['daatAndSkipTfIdf']['everything'][:10]

    # tf_idf = [item[1] for item in everything]


    # Map document IDs to the corresponding documents in food.json
    output_food = []

    # for doc_id in doc_ids:
    #     # Since the doc_id in the food list is based on the index (starting from 1), we map accordingly
    #     if doc_id - 1 < len(food_list):  # Check if doc_id exists in the food list
    #         output_food.append((doc_id,food_list[doc_id - 1]))
    #     else:
    #         output_food = []

    for doc_id, tf_idf in everything:
        if doc_id - 1 < len(food_list):  # Ensure valid index
            # Create a new object by updating the original food item
            food_item = food_list[doc_id - 1].copy()  # Copy the food object to avoid modifying the original
            food_item.update({
                "doc_id": doc_id,
                "tf_idf": tf_idf
            })
            output_food.append(food_item)

# # Save the output to output_food.json
#     with open('output_food.json', 'w') as output_file:
#         json.dump(output_food, output_file, indent=4)
#     print("Mapped output saved to output_food.json nad returned the result")
    return output_food

def classify(query):
    try:
        # Preparing messages for OpenAI

        messages = [
            {
                "role": "system",
                "content": "You are topic classifier, your topics are Food,Health,Economy,Education,Entertainment,Environment,Politics,Sports,Technology,Travel,General ,you can replay with multi topics separated by a comma remember only respond with the topic nothing else"
            },
            {
                "role": "system",
                "content": query
            },
        ]
        # print("sending thia to openai",messages)
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Call OpenAI's API

        client = OpenAI(api_key = api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",  # Replace with the desired model
            messages=messages
        )
        
        # Extract the reply
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return f"Error communicating with OpenAI {e}"


# Function to interact with OpenAI
def call_openai(m, dataset):
    try:
        # Preparing messages for OpenAI

        messages = [
            {
                "role": "system",
                "content": "You are assistant, your name is X. You will be given a dataset to Summarize. If you are not given a dataset, just engage in short conversation."
            },
            {
                "role": "system",
                "content": json.dumps(dataset)
            },
        ]
        messages.extend(m)
        print("sending thia to openai",messages)
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Call OpenAI's API

        client = OpenAI(api_key = api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",  # Replace with the desired model
            messages=messages
        )
        
        # Extract the reply
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return f"Error communicating with OpenAI {e}"


@app.route("/topic_classification", methods=['POST'])
def execute():
    query = request.json.get("query")
    replay = classify(query)
    return jsonify(replay)

@app.route("/execute_query", methods=['POST'])
def execute_query():
    print("hello")

    query = request.json.get("query")
    topics = request.json.get("topic")
    messages = request.json.get("messages")

    print("query",query);
    print("topic",topics);

    topic_values = [topic["value"] for topic in topics]  # Extract 'value' from each {value, label}
    print("topic values",topic_values);
   #   Run your indexer logic for each topic value
    combined_output = []  # To store results for each topic 
    dataset = []
    for topic_value in topic_values:
        if(topic_value != "General"):
            print("Processing topic:", topic_value)
            output = runner.run_queries(query, topic_value)  # Pass individual topic value
            combined_output.append(output)
            doc_ids = output['daatAndSkipTfIdf']['results']
            print(f"Output for {topic_value}:",doc_ids)
            d = saveOutput(output, topic_value)
            dataset.extend(d)
    openai_response = call_openai(messages, dataset)
    
    # Respond with OpenAI's output
    response = {
        "query_response": openai_response,
        "docReturned": dataset, # Include the original output_dict for reference
    }



    
    return jsonify(response)


if __name__ == "__main__":
    print("H")
    corpus = ["Food.json","Sports.json"]
            #   ,"Health.json","Economy.json","Education.json","Entertainment.json","Environment.json","Politics.json","Sports.json","Technology.json","Travel.json"]
    runner = ProjectRunner()
    for topic in corpus:
       runner.run_indexer_from_specific_field(topic)
    print("Hey")
    app.run(debug=True)

corpus = ["Food.json", "Economy.json","Education.json","Entertainment.json","Environment.json","Politics.json","Sports.json","Technology.json","Travel.json"]
runner = ProjectRunner()
for topic in corpus:
    runner.run_indexer_from_specific_field(topic)
print("Hey2")



@app.route("/execute2", methods=['POST'])
def execute_1():
    print("called2")
    return "calle2"



# def index(corpus):
#     projectRunner = ProjectRunner()
#     projectRunner.run_indexer_from_specific_field(corpus)

#     user_input = input("Please enter your query: ")
 
#     topic = "Food"
#     output = projectRunner.run_queries(user_input,topic)
#     doc_ids = output['daatAndSkipTfIdf']['results'][:10]
#     answer = call_openai(user_input, doc_ids)
#     print("Output:",doc_ids)
#     print("Answer:",answer)
#     saveOutput(output, topic)
# # index("Health.json")
# index("Food.json")

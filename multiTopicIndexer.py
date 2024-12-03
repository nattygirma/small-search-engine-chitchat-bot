from collections import defaultdict, OrderedDict
from math import sqrt, log
from indexer import Indexer

class MultiTopicIndexer:
    def __init__(self):
        """Initialize a dictionary to hold indexes for each topic."""
        self.topic_indexes = defaultdict(Indexer)

    def add_document(self, topic, doc_id, tokenized_document):
        """
        Add a document to the index of a specific topic.
        
        :param topic: The topic the document belongs to.
        :param doc_id: The document ID.
        :param tokenized_document: The list of tokens from the document.
        """
        # Retrieve the indexer for the topic, creating a new one if necessary
        indexer = self.topic_indexes[topic]
        indexer.generate_inverted_index(doc_id, tokenized_document)

    def get_topic_index(self, topic):
        """
        Retrieve the index for a specific topic.
        
        :param topic: The topic whose index is required.
        :return: The inverted index for the topic.
        """
        if topic in self.topic_indexes:
            return self.topic_indexes[topic].get_index()
        else:
            return None

    def process_index(self, topic):
        """
        Perform sorting, skip connection addition, and TF-IDF calculation for a specific topic.
    
        :param topic: The name of the topic whose index should be processed.
        """
        if topic in self.topic_indexes:
            indexer = self.topic_indexes[topic]
            indexer.sort_terms()
            indexer.add_skip_connections()
            indexer.calculate_tf_idf()
            # print(f"Indexing complete for topic '{topic}'")
        else:
            print(f"No index found for topic '{topic}'")
    def process_all_indexes(self):
        """Perform sorting, skip connection addition, and TF-IDF calculation for all topics."""
        for topic, indexer in self.topic_indexes.items():
            indexer.sort_terms()
            indexer.add_skip_connections()
            indexer.calculate_tf_idf()
# How to Run the Program
1. If you want to use your own python virtual environment, create your own environment with Python 3.9, and install the required libraries by running:

    Cloning SpanBERT git repository:
    ```
    git clone https://github.com/larakaracasu/SpanBERT
    cd SpanBERT
    ```
    
    Creating virtual environment:
    ```
    sudo apt-get -y update
    sudo apt install python3-venv
    python3 -m venv [preffered_name]
    ```
    
    Installing required libraries:
    ```
    pip3 install --upgrade google-api-python-client
    pip3 install beautifulsoup4
    pip3 install -U pip setuptools wheel
    pip3 install -U spacy
    python3 -m spacy download en_core_web_lg
    pip3 install -r requirements.txt
    pip install -q -U google-generativeai
    ```

2. Run the program using:
   ```
   python3 main.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>
   ```
   r: type of relation to extract (integer), t: threshold or confidence (float), q: seed query (string), k: number of top-k relations to extract (integer)

   Types of relations:
   1 = Schools Attended (person, organization), 
   2 = Work For (person, organization),
   3 = Live In (person, location/city/state or province/country),
   4 = Top Member Employees (organization, person)

   Example
   ```
   python3 main.py -spanbert <google api key> <google engine id> <google gemini api key> 2 0.7 "bill gates microsoft" 10
   ```

# Internal Design

## Libraries Used

- sys, ast, time: Used for basic Python operations, such as parsing command-line arguments, converting the response string into a python object, and timing operations to prevent exeeding gemini query limit.
- requests: Handles HTTP requests to retrieve web pages.
- BeautifulSoup: Parses HTML content and cleans up the text from the webpages.
- spacy: Used for natural language processing tasks, particularly for splitting texts into sentences and extracting named entities.
- spanbert, spacy_help_functions: Provided custom modules for running the SpanBERT model, which predicts relationships between pairs of named entities in a sentence.
- google.generativeai, googleapiclient.discovery: Interfaces for Google's APIs, used for relationship extraction via the Gemini API and Custom Search Engine respectively.
- collections.defaultdict: Utilized for efficiently handling dictionaries with default values, particularly for storing extracted relations and their confidences.

## Main Components (High Level Overview)

- google_search(): Uses the Google Custom Search Engine to search for pages based on a query, returning the top results.
- extract_text_from_url(): Downloads webpage content via its URL and uses BeautifulSoup to extract and clean the plain text from the HTML.
- extract_relations(): Depending on the method chosen (SpanBERT or Gemini), this function uses either SpanBERT model (with named entity pairs from spaCy) or sends sentences to the Gemini API (with pre-screening with spaCy) for relation extraction. It manages the collection of extracted relation tuples, adhering to specified thresholds for inclusion and avoiding duplicates.
- main block: Performs the overall Iterative Set Expansion (ISE) algorithm for relation extraction. It begins by parsing command-line arguments to set up initial configurations. It then manages the iterative search process, invoking the above functions to handle text extraction, entity recognition, and relationship extraction for each URL. The block also manages the search iteration, updating the query based on the results obtained and ensuring the desired number of relation tuples (k) is eventually reached, unless the available URLs are exhausted first.

# Implementation of Iterative Set Expansion (ISE) algorithm

## URL Processing and Text Extraction

- Skips already processed URLs to avoid redundancy.
- For each new URL, attempt to retrieve the webpage using the requests library. If the page is successfully retrieved, it employs BeautifulSoup to remove unnecessary tags (like script, style) and extracts plain text.
- If the extracted text exceeds 10,000 characters, it's truncated to ensure efficiency.

## Relation Extraction 
- The text is processed by spaCy to split it into sentences and extract named entities.
- Depending on the command-line arguments, choose between using SpanBERT and Gemini for relation extraction.
- SpanBERT: It uses pre-defined mappings between entity types to filter out relevant entity pairs for the specified relation type. The filtered pairs are then passed to SpanBERT, which returns predictions and confidences. Tuples meeting the confidence threshold are added to set X. Duplicates are prevented, unless the new tuple has higher confidence.
- Gemini: For efficient use of API calls, first check if the sentence has a "possibility" to extract relationship (contains the required entity types for the relation). Then construct a prompt for the Gemini API based on the required entity types and the sentence. We then parse the Gemini response to extract relation tuples, which are all added to set X with a hardcoded confidence of 1.0.

## Iteration and Output
- Our script iterates through this process, updating the set X with new relation tuples and their confidences until it either meets the desired number of tuples or exhausts the list of URLs without reaching this number.
- For SpanBERT, it sorts and prints the extracted relations by confidence. For Gemini, it simply prints all extracted relations without sorting since all confidences are hardcoded to 1.0.
- If the set X reaches the desired number of tuples or if no new tuples can be generated, the script ends. Otherwise, it generates a new query based on unused tuples from X and continues to the next iteration.


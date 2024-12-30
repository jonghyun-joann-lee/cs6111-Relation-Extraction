import sys
import ast
import time
import spacy
import requests
from spanbert import SpanBERT
from bs4 import BeautifulSoup
import google.generativeai as genai
from collections import defaultdict
from googleapiclient.discovery import build
from spacy_help_functions import create_entity_pairs

# Function to perform Google search using the Custom Search Engine
def google_search(query, google_search_API, engine_ID, **kwargs):
    service = build("customsearch", "v1", developerKey=google_search_API)
    result = service.cse().list(q=query, cx=engine_ID, **kwargs).execute()
    # If there is no query result
    if not "items" in result:
        return []
    return result['items']

# Function to extract text from the given URL using requests and BeautifulSoup
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                script_or_style.decompose()
                
            plain_text = soup.get_text(separator=' ', strip=True)
            return plain_text
    except Exception as e:
        print(f"Error retrieving {url}: {e}")
    return None

# Function to extract relations from a sentence using either SpanBERT or Gemini
def extract_relations(sentence, spanbert, r, threshold, X, method, bert2spacy, entity_relation_mapping, examples_and_description):
    sent_overall = 0 # Keep track of number of extracted relations
    sent_added = 0 # Keep track of number of added relations among extracted relations
    
    # Prepare entities for extraction
    subj_type, obj_type = entity_relation_mapping[r][0]
    entities_of_interest = [subj_type] + obj_type if isinstance(obj_type, list) else [subj_type, obj_type]

    # Process sentence for relation extraction
    if method == "spanbert":
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        candidates = []
        for ep in entity_pairs:
            candidates.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            candidates.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        
        # Only keep entity pairs that have the required entity types for the relation of interest
        valid_candidates = [c for c in candidates if c["subj"][1] == subj_type and c["obj"][1] in obj_type]
        
        # Run SpanBERT over valid entity pairs if there exist
        if valid_candidates:
            preds = spanbert.predict(valid_candidates) 

            for cand, pred in list(zip(valid_candidates, preds)):
                relation = pred[0]
                if relation == entity_relation_mapping[r][1]:
                    sent_overall += 1
                    print("\n\t\t=== Extracted Relation ===")
                    print(f"\tInput tokens: {cand['tokens']}")
                    subj = cand["subj"][0]
                    obj = cand["obj"][0]
                    confidence = pred[1]
                    print(f"\t\tOutput Confidence: {confidence:.8f} ; Subject: {subj} ; Object: {obj} ;")
                    if confidence > threshold:
                        if X[(subj, relation, obj)] < confidence:
                            print("\t\tAdding to set of extracted relations")
                            X[(subj, relation, obj)] = confidence
                            sent_added += 1
                        else:
                            print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
                    else:
                        print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
                    print("\t\t==========")
    
    if method == "gemini":
        subj_type_spacy = bert2spacy[subj_type]
        if isinstance(obj_type, list):
            obj_type_spacy = [bert2spacy[b] for b in obj_type]
            prompt_obj_type = obj_type[0] # Only 'LOCATION' for relation Live_In
        else:
            obj_type_spacy = bert2spacy[obj_type]
            prompt_obj_type = obj_type
        found_subj_type, found_obj_type = False, False 
        ents = sentence.ents
        for ent in ents: # Check if required entity types are found in sentence
            if ent.label_ == subj_type_spacy:
                found_subj_type = True
            if ent.label_ in obj_type_spacy:
                found_obj_type = True
        
        # Only ask Gemini if all required entity types are in the sentence
        if found_subj_type and found_obj_type:
            model = genai.GenerativeModel("gemini-pro") # Initialize model
            generation_config = genai.types.GenerationConfig( # Configure model with parameters
                max_output_tokens = 100,
                temperature = 0.2,
                top_p = 1,
                top_k = 32
            )
            # Query Gemini
            prompt = f"""
                    Given a sentence, perform named entity recognition to identify PERSON, ORGANIZATION, and LOCATION. Then, determine if any two of the identified entities are involved in the relation {entity_relation_mapping[r][2]}, meaning {examples_and_description[r][1]}. If so, extract these relations as tuples.

                    Sentence: {sentence.text}

                    Expected format of tuples: ({entity_relation_mapping[r][0][0]}, {entity_relation_mapping[r][2]}, {prompt_obj_type})
                    Example tuple: {examples_and_description[r][0]}

                    Process the above sentence and report all extracted relations as a list of tuples. If you cannot find any, then return None."""
            response = model.generate_content(prompt, generation_config=generation_config) # Generate a response

            # Process Gemini's response
            try: 
                response_text = response.text
                response_eval = ast.literal_eval(response_text) # Convert the response string into a Python object 

                if response_eval: # A list of tuples of (subj, relation, obj)
                    for relation_tuple in response_eval:
                        sent_overall += 1
                        print("\n\t\t=== Extracted Relation ===")
                        print(f"\t\tSentence: {sentence.text}")
                        print(f"\t\tSubject: {relation_tuple[0]} ; Object: {relation_tuple[2]} ;")
                        if relation_tuple in X:
                            print("\t\tDuplicate. Ignoring this.")
                        else:
                            print("\t\tAdding to set of extracted relations")
                            X[relation_tuple] = 1
                            sent_added += 1
                        print("\t\t==========")
            except:
                pass
            
            time.sleep(0.5) # sleep to prevent exhausting gemini query limits
                
    return X, sent_overall, sent_added


# Main Block
if __name__ == "__main__":

    # Collect arguments and configurations
    method = sys.argv[1][1:]
    google_search_API = sys.argv[2]
    engine_ID = sys.argv[3]
    gemini_API = sys.argv[4]
    r = int(sys.argv[5])
    threshold = float(sys.argv[6])
    query = sys.argv[7]
    k =int(sys.argv[8])
    
    # Initialize necessary components and variables
    iteration = 0
    X = defaultdict(int) # Dict of extracted relations with their confidence
    seen_urls = set()
    used_queries = {query.lower} # Set of used query strings where order of words matters
    entity_relation_mapping = { # Used in extract_relations() and main block
        1: (('PERSON', 'ORGANIZATION'), 'per:schools_attended', 'Schools_Attended'),
        2: (('PERSON', 'ORGANIZATION'), 'per:employee_of', 'Work_For'),
        3: (('PERSON', ['LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY']), 'per:cities_of_residence', 'Live_In'),
        4: (('ORGANIZATION', 'PERSON'), 'org:top_members/employees', 'Top_Member_Employees')
    }
    examples_and_description = { # Used in extract_relations()
        1: (("Jeff Bezos", "Schools_Attended", "Princeton University"), "a PERSON has been enrolled as a student at an ORGANIZATION (any school including university, college, high school, etc.). Look for phrases like graduate of, student in, studied at, etc"),
        2: (("Alec Radford", "Work_For", "OpenAI"), "a PERSON has been an employee or member of an ORGANIZATION"),
        3: (("Mariah Carey", "Live_In", "New York City"), "a PERSON has lived in a LOCATION (at the level of city, town, village, state, province, or country) for a certain time"),
        4: (("Nvidia", "Top_Member_Employees", "Jensen Huang"), "a PERSON is in high-level, leading position (e.g., founder, co-chair, chief officer, president, vice president, board member, etc.) at an ORGANIZATION")
    }
    bert2spacy = { # Used in extract_relations()
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
    }

    # Load pre-trained SpanBERT model if specified in the method
    if method == "spanbert":
        spanbert = SpanBERT("./pretrained_spanbert")  
    else:
        spanbert = None
   
    # Print Configurations
    print("\n\n---")
    print("Parameters:")
    print(f"Client key\t= {google_search_API}")
    print(f"Engine key\t= {engine_ID}")
    print(f"Gemini key\t= {gemini_API}")
    print(f"Method\t\t= {method}")
    print(f"Relation\t= {entity_relation_mapping[r][2]}")
    print(f"Threshold\t= {threshold}")
    print(f"Query\t\t= {query}")
    print(f"# of Tuples\t= {k}")
    print("Loading necessary libraries; This should take a minute or so ...")
    
    # Configure Gemini API if specified in the method
    if method == "gemini":
        genai.configure(api_key=gemini_API)
    # Initialize spaCy for named entity recognition
    nlp = spacy.load("en_core_web_lg") 

    # Main loop for processing      
    while True: 
        print(f"=========== Iteration: {iteration} - Query: {query} ===========")
        
        # Query Google Custom Search Engine to obtain the URLs for the top-10 webpages for query
        results = google_search(query, google_search_API, engine_ID, num=10)
        
        for i, result in enumerate(results):
            url = result["link"]
            print(f"\nURL ( {i+1} / {len(results)}): {url}")
            
            # Skip already seen URL
            if url in seen_urls:
                print("URL already seen. Continuing.")
                continue
            seen_urls.add(url)
            
            print("\t Fetching text from url ...")
            # If file format is not in HTML, skip the document
            if result.get('fileFormat'):
                print("Unable to fetch URL. Continuing.")
                continue
            else:
                # Extract the actual plain text from the webpage using Beautiful Soup.
                url_content = extract_text_from_url(url)
            
            # If there is no url_content, skip the document
            if not url_content: 
                print("Unable to fetch URL. Continuing.")
                continue
            
            # If the resulting plain text is longer than 10,000 characters, 
            # truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
            if len(url_content) > 10000: 
                print(f"\t Trimming webpage content from {len(url_content)} to 10000 characters")
                url_content = url_content[:10000]
            
            print(f"\t Webpage length (num characters): {len(url_content)}")
            print(f"\t Annotating the webpage using spacy...")

            doc = nlp(url_content) # Use spaCy to split the text into sentences and extract named entities
            sentences = list(doc.sents)
            print(f"\t Extracted {len(sentences)} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            page_extracted_sent = 0
            page_overall = 0
            page_added = 0
            
            for i, sentence in enumerate(sentences):
                X, sent_overall, sent_added = extract_relations(sentence, spanbert, r, threshold, X, method, bert2spacy, entity_relation_mapping, examples_and_description)
                if sent_overall > 0:
                    page_extracted_sent += 1
                page_overall += sent_overall
                page_added += sent_added
                
                # Report progress every fifth sentence
                if (i+1) % 5 == 0:
                    print(f"\t Processed {i+1} / {len(sentences)} sentences")
            
            # Report output of current URL
            print(f"\t Extracted annotations for {page_extracted_sent} out of total {len(sentences)} sentences")
            print(f"\t Relations extracted from this website: {page_added} (Overall: {page_overall})")

        # Report ouput of current iteration
        # If spanbert, sort the relations by confidence in descending order and report them 
        if method == "spanbert":
            sorted_X = sorted(X.items(), key=lambda x: x[1], reverse=True)
            print(f"================== ALL RELATIONS for {entity_relation_mapping[r][1]} ( {len(X)} ) =================")
            for relation_tuple, confidence in sorted_X:
                print(f"Confidence: {confidence:.8f}\t\t| Subject: {relation_tuple[0]}\t\t| Object: {relation_tuple[2]}")
        if method == "gemini":
            print(f"================== ALL RELATIONS for {entity_relation_mapping[r][2]} ( {len(X)} ) =================")
            for relation_tuple in X.keys():
                print(f"Subject: {relation_tuple[0]}\t\t| Object: {relation_tuple[2]}")
        
        # Terminate if we have more than k realations
        # Else, generate new query and continue to new iteration
        if len(X) >= k:
            print(f"Total # of iterations = {iteration+1}")
            break  
        else:
            new_query_generated = False
            # Select a tuple from X that has not been used for querying yet and has the highest confidence
            if method == "spanbert":
                items = sorted_X
            else:
                items = X.items()
            for relation_tuple, confidence in items:
                new_query = relation_tuple[0] + ' ' + relation_tuple[2] # Generate new query "subj + obj"
                if new_query.lower() not in used_queries: # Case-insensitive, exact ordering match
                    used_queries.add(new_query.lower())
                    query = new_query
                    new_query_generated = True
                    break
            # If no such tuple exists, then stop
            if not new_query_generated:
                print(f"ISE has stalled before retrieving {k} high-confidence tuples.")
                break
        
        iteration += 1
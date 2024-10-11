import torch
import transformers

from graphs.contriever_graph import LLaMAContrieverGraph, ContrieverGraph
from agents.llama_agent import LLaMAagent
from agents.parent_agent import GPTagent
from utils.utils import Logger
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD
import re

# Параметры конфигурации
graph_model, qa_model = "gpt-4o-mini", "gpt-4o-mini"

YOUR_OPENAI_API_KEY = ""

def get_answer(agent, question, subgraph, episodic):
    prompt = f'''Your task is to answer the following question: "{question}"

Relevant facts from your memory: {subgraph}

Relevant texts from your memory: {episodic}

Answer the question "{question}" with a Chain of Thoughts in the following format:
"CoT: your chain of thoughts
Direct answer: your direct answer to the question"
Direct answer must be concrete and must not contain alternatives, descriptions, or reasoning.
Write "Unknown" if you have doubts.
Do not write anything except the answer in the given format.

Your answer: '''
    return agent.generate(prompt)


def load_setup(graph_model, qa_model):
    # Загрузка моделей и агентов
    if "llama" in graph_model or "llama" in qa_model:
        pipeline = transformers.pipeline(
            "text-generation",
            model="Undi95/Meta-Llama-3-70B-Instruct-hf",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    if "llama" in graph_model:
        graph = LLaMAContrieverGraph("", "You are a helpful assistant", "", pipeline, "cuda")
        agent_items = LLaMAagent("You are a helpful assistant", pipeline)
    else:
        graph = ContrieverGraph(graph_model, "You are a helpful assistant",
                                YOUR_OPENAI_API_KEY)
        agent_items = GPTagent(graph_model, "You are a helpful assistant",
                               YOUR_OPENAI_API_KEY)

    if "llama" in qa_model:
        agent_qa = LLaMAagent("You are a helpful assistant", pipeline)
    else:
        agent_qa = GPTagent(qa_model, "You are a helpful assistant",
                            YOUR_OPENAI_API_KEY)

    return agent_items, agent_qa, graph


def extract_ontologies_and_create_rdf(text):
    # Initialize the RDF graph
    g = Graph()

    # Namespaces
    EX = Namespace('http://example.org/')
    g.bind('ex', EX)
    g.bind('rdf', RDF)
    g.bind('rdfs', RDFS)
    g.bind('xsd', XSD)

    # Define Classes
    person_class = EX.Person
    location_class = EX.Location
    movement_event_class = EX.MovementEvent

    g.add((person_class, RDF.type, RDFS.Class))
    g.add((location_class, RDF.type, RDFS.Class))
    g.add((movement_event_class, RDF.type, RDFS.Class))

    # Define Properties
    has_name = EX.hasName
    performed = EX.performed
    action_prop = EX.action
    destination_prop = EX.destination
    sequence_prop = EX.sequence

    g.add((has_name, RDF.type, RDF.Property))
    g.add((has_name, RDFS.domain, person_class))
    g.add((has_name, RDFS.domain, location_class))
    g.add((has_name, RDFS.range, RDFS.Literal))

    g.add((performed, RDF.type, RDF.Property))
    g.add((performed, RDFS.domain, person_class))
    g.add((performed, RDFS.range, movement_event_class))

    g.add((action_prop, RDF.type, RDF.Property))
    g.add((action_prop, RDFS.domain, movement_event_class))
    g.add((action_prop, RDFS.range, RDFS.Literal))

    g.add((destination_prop, RDF.type, RDF.Property))
    g.add((destination_prop, RDFS.domain, movement_event_class))
    g.add((destination_prop, RDFS.range, location_class))

    g.add((sequence_prop, RDF.type, RDF.Property))
    g.add((sequence_prop, RDFS.domain, movement_event_class))
    g.add((sequence_prop, RDFS.range, RDFS.Literal))

    # Extract sentences from text
    sentences = re.split(r'\.\s*', text.strip())
    sentences = [s for s in sentences if s]  # Remove empty strings

    persons = {}
    locations = {}
    timestamp = 1

    for sentence in sentences:
        # Extract person, action, destination
        match = re.match(r'(\w+)\s+(.*?)\s+to the\s+(\w+)', sentence)
        if match:
            person_name = match.group(1)
            action = match.group(2).strip()
            destination_name = match.group(3)

            # Create person instance
            person_uri = EX[person_name]
            if person_uri not in persons:
                g.add((person_uri, RDF.type, person_class))
                g.add((person_uri, has_name, Literal(person_name)))
                persons[person_uri] = person_name

            # Create location instance
            location_uri = EX[destination_name.capitalize()]
            if location_uri not in locations:
                g.add((location_uri, RDF.type, location_class))
                g.add((location_uri, has_name, Literal(destination_name.lower())))
                locations[location_uri] = destination_name.lower()

            # Create movement event instance
            movement_event_uri = EX[f'MovementEvent{timestamp}']
            g.add((movement_event_uri, RDF.type, movement_event_class))
            g.add((movement_event_uri, action_prop, Literal(action)))
            g.add((movement_event_uri, destination_prop, location_uri))
            g.add((movement_event_uri, sequence_prop, Literal(str(timestamp), datatype=XSD.integer)))

            # Link person to movement event
            g.add((person_uri, performed, movement_event_uri))

            timestamp += 1

    return g


def run():
    # Инициализация логирования
    log = Logger(log_path)

    # Загрузка агентов и графа
    agent_items, agent_qa, graph = load_setup(graph_model, qa_model)

    # Определяем текст и вопрос
    text = 'John travelled to the hallway. John travelled to the office. Mary went back to the kitchen. ' \
           'Mary went back to the bathroom. Daniel journeyed to the hallway. ' \
           'Daniel moved to the kitchen.'
    question = 'Where is Mary?'
    true_answer = 'Mary is in the bathroom.'

    rdf_graph = extract_ontologies_and_create_rdf(text)

    print(rdf_graph.serialize(format='turtle').encode('utf-8'))

    # Define the query
    query = '''
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?location WHERE {
      ?person rdfs:label "Mary" .
      ?person dbo:location ?location .
    }
     '''

    # # Execute the query
    results = rdf_graph.query(query)

    # Print the results
    print(results)

    # q = graph.create_sparql(question)
    #
    # print(q)
    #
    # results2 = rdf_graph.query(q)
    #
    # # Print the results
    # for row in results2:
    #     print(f"{row}")


if __name__ == "__main__":
    run()

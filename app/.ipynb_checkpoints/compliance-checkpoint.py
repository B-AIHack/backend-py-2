import asyncio
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from enum import Enum
from langchain_community.llms import Ollama
from transformers import pipeline
import re
import unicodedata

classifier = pipeline("zero-shot-classification",  model="joeddav/xlm-roberta-large-xnli")


def clean_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"[^\w\s,.–—:;!?()«»\"%№/-]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<=\w)\n(?=\w)", "", text)
    text = re.sub(r"\b[а-яА-Яa-zA-Z]\b", "", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[ \t\u00A0]{2,}", " ", text)
    text = text.strip()
    return text


class ContractType(Enum):
    PRODUCTS = "Товарные договоры"
    SERVICES = "Договоры на оказание услуг"
    LOANS = "Договоры займа и кредита"
    INVESTMENTS = "Инвестиционные договоры"


def get_contract_type(contract_text: str) -> ContractType:
    labels_new = [member.value for member in ContractType]
    result_new = classifier(clean_ocr_text(contract_text), labels_new)
    max_score_index = result_new['scores'].index(max(result_new['scores']))
    best_label = result_new['labels'][max_score_index]
    return ContractType(best_label)


rule_check_prompt = PromptTemplate.from_template("""
Ты ассистент валютного контроля. Твоя задача — проверить текст валютного договора на соответствие только одному правилу валютного контроля.
Правило: {{rule}}
ID правила: {{id}}
Ответ только на русском языке.

Текст договора:
{{contract_text}}

Договор нарушает правило?
Если ДА, верни ответ только в JSON формате:
{{
  "violation": true,
  "rule_id": "{id}",
  "matched_text": <текст, где есть нарушение>,
}}

Если НЕТ, верни ответ только в JSON формате:
{{ "violation": false }}
""")

rules = [
  {
    "id": "R001",
    "rule": "В договоре должен быть указан срок репатриации валютной выручки.",
    "applies_to": [ContractType.PRODUCTS],
    "references": ["Правила экспортно‑импортного валютного контроля ПНБ РК от 29.09.2023 № 78 – определение срока репатриации и порядок контроля"] 
  },
  {
    "id": "R002",
    "rule": "В договоре должны быть указаны банковские реквизиты сторон.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Закон РК об валютном регулировании и контроле, ст. 7–8 – реквизиты банков резидентов/нерезидентов"]
  },
  {
    "id": "R004",
    "rule": "Валютные операции между резидентами РК запрещены вне внутреннего валютного рынка.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Закон РК об валютном регулировании и контроле, ст. 6–7 (валютные операции между резидентами/нерезидентами)"]
  },
  {
    "id": "R005",
    "rule": "В договоре должна быть указана сумма сделки.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["ПНБ – сумма договора используется при определении порога учета и репатриации"]
  },
  {
    "id": "R006",
    "rule": "Если сумма договора > 10 млн ₸ (≈50 000 USD) — договор должен быть зарегистрирован в НБ РК (учётный номер / паспорт сделки) до начала исполнения.",
    "applies_to": [ContractType.PRODUCTS, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Закон РК «О валютном регулировании и…», ст. 9; Правила экспортно‑импортного контроля, п. 49 и след."]
  },
  {
    "id": "R009",
    "rule": "Дата договора должна совпадать во всех языковых версиях.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Комплаенс‑правила банков по верификации и переводу документов"] 
  },
  {
    "id": "R011",
    "rule": "Для товарных договоров должен быть указан код ТН ВЭД.",
    "applies_to": [ContractType.PRODUCTS],
    "references": ["Таможенный кодекс ЕАЭС – классификация товаров; банковский валютный контроль"] 
  },
  {
    "id": "R012",
    "rule": "Если договор на иностранном языке — обязателен перевод на русский или казахский.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Правила экспортно‑импортного валютного контроля ПНБ; Закон РК «О валютном регулировании…», ст. 9"]
  },
  {
    "id": "R013",
    "rule": "В договоре должен быть указан номер договора.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Комплаенс‑требования банковского документооборота"] 
  },
  {
    "id": "R014",
    "rule": "В договоре должна быть указана валюта расчётов.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Закон РК «О валютном регулировании…», ст. 7–8"]
  },
  {
    "id": "R015",
    "rule": "Условия и порядок оплаты (валюта, сроки, реквизиты, банк‑корреспондент) должны быть четко прописаны.",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Закон РК «О валютном регулировании…», ст. 7; Правила ПНБ по валютным операциям"]
  },
  {
    "id": "R016",
    "rule": "В товарных договорах — должны быть условия поставки (Incoterms или эквивалент).",
    "applies_to": [ContractType.PRODUCTS],
    "references": ["Таможенные и логистические нормы ЕАЭС; комплаенс‑требования банков"] 
  },
  {
    "id": "R017",
    "rule": "В договоре должны быть указаны сроки исполнения обязательств (поставка, услуги, возврат займа и т.д.).",
    "applies_to": [ContractType.PRODUCTS, ContractType.SERVICES, ContractType.LOANS, ContractType.INVESTMENTS],
    "references": ["Закон РК «О гражданских обязательствах»; комплаенс‑правила"] 
  },
  {
    "id": "R018",
    "rule": "Если это договор на услуги — указывать объект, объем, срок и результат.",
    "applies_to": [ContractType.SERVICES],
    "references": ["Гражданский кодекс РК; банковские комплаенс‑правила"] 
  },
  {
    "id": "R019",
    "rule": "Если это займ/кредит — указывать сумму, процент, срок, способ возврата.",
    "applies_to": [ContractType.LOANS],
    "references": ["Закон РК «О займах и кредитах»; Регламент ПНБ"] 
  },
  {
    "id": "R020",
    "rule": "Инвестиционный договор: указать обязательства сторон, сроки, форму внесения инвестиций.",
    "applies_to": [ContractType.INVESTMENTS],
    "references": ["Закон РК «Об инвестициях»; валютное законодательство"] 
  },
  {
    "id": "S001",
    "rule": "В договоре на оказание услуг должен быть четко определен предмет договора (что именно предоставляется).",
    "applies_to": [ContractType.SERVICES],
    "references": ["Гражданский кодекс РК, ст. 384; Положения банковского комплаенса"]
  },
  {
    "id": "S002",
    "rule": "Должны быть указаны сроки начала и окончания оказания услуг.",
    "applies_to": [ContractType.SERVICES],
    "references": ["ГК РК, ст. 386; требования комплаенс-служб банков"]
  },
  {
    "id": "S003",
    "rule": "В договоре должен быть указан объем или формат оказания услуг (единицы, часы, этапы и пр.).",
    "applies_to": [ContractType.SERVICES],
    "references": ["ГК РК, ст. 387"]
  },
  {
    "id": "S008",
    "rule": "Договор должен содержать результат оказания услуг (отчет, акт, продукт и пр.).",
    "applies_to": [ContractType.SERVICES],
    "references": ["ГК РК, ст. 388"]
  },
  {
    "id": "L001",
    "rule": "В договоре займа или кредита должна быть указана процентная ставка или условие её отсутствия.",
    "applies_to": [ContractType.LOANS],
    "references": ["Гражданский кодекс РК, ст. 715, 716"]
  },
  {
    "id": "L003",
    "rule": "В договоре должен быть указан график возврата займа (дата/этапы, сумма, периодичность).",
    "applies_to": [ContractType.LOANS],
    "references": ["Гражданский кодекс РК, ст. 717; Комплаенс-требования банков по контролю валютных операций"]
  }
]

# Output parser
class SimpleJSONParser(BaseOutputParser):
    def parse(self, text: str):
        import json, re
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {"violation": False}
        except Exception:
            return {"violation": False}

parser = SimpleJSONParser()


class ContractState(dict):
    contract_text: str
    violations: list


def make_agent_node(rule):
    chain = LLMChain(prompt=rule_check_prompt, llm=Ollama(model="llama3:70b-instruct-q2_K"), output_parser=parser)
    def node(state: ContractState):
        result = chain.run(
            contract_text=state["contract_text"],
            rule=rule["rule"],
            id=rule["id"]
        )
        if result.get("violation"):
            state["violations"].append(result)
        return state
    return node

# Build the LangGraph graph
def build_graph(rules):
    graph = StateGraph(ContractState)

    # Add agent nodes
    for rule in rules:
        graph.add_node(f"rule_{rule['id']}", make_agent_node(rule))

    # Define start node
    graph.set_entry_point(f"rule_{rules[0]['id']}")

    # Chain agents sequentially OR run in parallel
    for i, rule in enumerate(rules[:-1]):
        next_rule = rules[i + 1]
        graph.add_edge(f"rule_{rule['id']}", f"rule_{next_rule['id']}")

    # Final edge to END
    graph.add_edge(f"rule_{rules[-1]['id']}", END)

    return graph.compile()


def compliance_validation(contract_text: str):

    contract_type = get_contract_type(contract_text)
    
    initial_state = {
        "contract_text": contract_text,
        "violations": []
    }
    product_rules = [
        rule for rule in rules
        if contract_type.value in [
            x.value if isinstance(x, Enum) else x for x in rule["applies_to"]
        ]
    ]
    graph = build_graph(product_rules)
    final_state = graph.invoke(initial_state)

    # Show violations
    # for v in final_state['violations']:
        # print(f"❌ Rule {v['rule_id']} violated:\nReason: {v['reason']}\nMatched: {v['matched_text']}\n")

    if not final_state:
        return []
        
    filtered_results = []
    for rule in rules:
        for violation in final_state['violations']:
            if violation['rule_id'] == rule['id']:
                filtered_results.append({
                    'rule_id': rule['id'],
                    'rule': rule['rule'],
                    'matched_text': violation['matched_text'],
                    'references': rule['references']
                })
    return filtered_results
    


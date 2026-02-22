import json
import random
from pathlib import Path

random.seed(42)

OUT = Path("data/nodes.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

UNIVERSITIES = [
    "Saint Louis University",
    "Washington University in St. Louis",
    "Harvard University",
    "MIT",
    "Stanford University",
    "UC Berkeley",
    "NYU",
    "Columbia University",
    "University of Chicago",
    "University of Toronto",
    "University of British Columbia",
    "University of Oxford",
    "University of Cambridge",
    "Imperial College London",
    "UCL",
    "ETH Zurich",
    "TU Munich",
    "Sorbonne University",
    "IIT Delhi",
    "IIT Bombay",
    "National University of Singapore",
    "Nanyang Technological University",
    "Tsinghua University",
    "University of Tokyo",
    "University of Melbourne",
    "University of Sydney",
    "KAUST",
    "University of Cape Town",
]

DOMAINS = ["AI/ML", "Business", "Coding", "Design", "Healthcare", "Robotics", "Data Science", "Cybersecurity"]
ROLES = ["Research Assistant", "Graduate Student", "PhD Student", "AI Engineer", "Data Scientist", "Product Manager", "UX Designer", "Robotics Engineer", "Security Analyst", "Founder"]
CLUBS = ["AI Society", "Data Science Club", "Robotics Club", "Entrepreneurship Club", "Design Collective", "Cybersecurity Club"]
EVENTS = ["AI Meetup", "Hackathon Prep", "Startup Pitch Night", "Research Showcase", "Career Fair", "Workshop: ML Basics"]

FIRST = ["Aarav", "Diya", "Isha", "Rohan", "Kabir", "Nora", "Maya", "Ethan", "Liam", "Olivia", "Noah", "Sophia"]
LAST  = ["Patel", "Shah", "Mehta", "Johnson", "Brown", "Garcia", "Kim", "Chen", "Singh", "Khan", "Taylor", "Martin"]

def make_person(i, uni):
    fn = random.choice(FIRST)
    ln = random.choice(LAST)
    dom = random.choice(DOMAINS)
    role = random.choice(ROLES)
    tags = random.sample(DOMAINS, k=3)
    return {
        "id": f"p_{i}",
        "title": f"{fn} {ln}",
        "type": "person",
        "domain": dom,
        "university": uni,
        "tags": tags + [role],
        "description": f"{role}. Interested in {', '.join(tags)}."
    }

def make_club(i, uni):
    dom = random.choice(DOMAINS)
    name = random.choice(CLUBS)
    tags = random.sample(DOMAINS, k=3)
    return {
        "id": f"c_{i}",
        "title": name,
        "type": "club",
        "domain": dom,
        "university": uni,
        "tags": tags + ["community", "students"],
        "description": f"A student community for {dom}. Weekly sessions and projects."
    }

def make_event(i, uni):
    dom = random.choice(DOMAINS)
    name = random.choice(EVENTS)
    tags = random.sample(DOMAINS, k=3)
    return {
        "id": f"e_{i}",
        "title": name,
        "type": "event",
        "domain": dom,
        "university": uni,
        "tags": tags + ["networking"],
        "description": f"An event focused on {dom}. Meet people and explore opportunities."
    }

def main():
    nodes = []
    pid = cid = eid = 0

    for uni in UNIVERSITIES:
        for _ in range(8):
            nodes.append(make_person(pid, uni)); pid += 1
        for _ in range(3):
            nodes.append(make_club(cid, uni)); cid += 1
        for _ in range(3):
            nodes.append(make_event(eid, uni)); eid += 1

    OUT.write_text(json.dumps(nodes, indent=2), encoding="utf-8")
    print(f"Generated {OUT} with {len(nodes)} nodes (~{len(nodes)//len(UNIVERSITIES)} per university)")

if __name__ == "__main__":
    main()
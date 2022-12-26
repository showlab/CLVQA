import json
import tqdm
import sys
import random
import numpy as np
# from pattern.text.en import singularize
import inflect

p = inflect.engine()
singularize = p.singular_noun


class GenerateSceneGraph:
    def __init__(self, meta_path="meta_sg_final.json", kb_path="knowledge_graph.json") -> None:
        meta_data = json.load(open(meta_path))
        self.meta_data = meta_data
        self.sg = meta_data['sg']
        self.sort_sg_count = meta_data['sort_sg_count']
        self.name2rel = meta_data['name2rel']
        self.att2rel = meta_data['att2rel']
        self.rel2sg = {}
        for k, v in self.meta_data['sg']['knowledge'].items():
            self.rel2sg[" ".join(v)] = v
        self.att_type2rel = self.get_att_type2rel()
        self.kb = json.load(open(kb_path))
        self.name2kg = {}

    def generate_sg(self, stage, sample_number=1, rel_number=1):
        meta = self.meta_data
        generated_sg = []
        if stage in ['object']:
            for i in tqdm.tqdm(range(sample_number)):
                gsg = []
                for _ in range(rel_number):
                    rel = self.sg[stage][self.random_select(
                        self.sort_sg_count[stage])[0]]
                    gsg.append(" ".join(rel).strip())
                generated_sg.append(gsg)

        elif stage in ['attribute']:
            sort_sg_count = meta['sort_sg_count']
            for i in tqdm.tqdm(range(int(sample_number * 1.5))):
                gsg = []
                for _ in range(rel_number * 2):
                    rel = self.random_select(sort_sg_count[stage])[0]
                    if ";" not in rel:
                        continue
                    rel = self.sg[stage][rel]
                    gsg.append(" ".join(rel).strip())
                    if len(gsg) == rel_number:
                        break

                if len(gsg) < rel_number:
                    continue

                generated_sg.append(gsg)
                if len(generated_sg) == sample_number:
                    break

        elif stage in ['relation']:
            sg_rel = meta['sg']['relation']
            for i in tqdm.tqdm(range(sample_number * 8)):
                gsg = []
                nodes = []
                adjecent = []
                rel_number = 3
                flag = True
                for j in range(rel_number):
                    if j == 0:
                        rel = sg_rel[self.random_select(
                            self.sort_sg_count[stage])[0]]
                    else:
                        rel = sg_rel[self.random_select(adjecent)[0]]

                    if j == 2:
                        if (' to ' in gsg[0] and ' to ' in gsg[1]) or (gsg[0] == gsg[1]) or (
                                len(gsg[1].split()) == 1 or len(gsg[0].split()) == 1):
                            flag = False
                            break
                    add_nodes = [rel[-1]] if len(rel) < 3 else [rel[0], rel[2]]

                    add_nodes = [k for k in add_nodes if k not in nodes]

                    for n in add_nodes:
                        adjecent += self.name2rel[stage][n]

                    nodes += add_nodes
                    gsg.append(" ".join(rel).strip())

                if flag:
                    generated_sg.append(gsg[:2])

                if len(generated_sg) == sample_number:
                    break

        elif stage in ['knowledge']:
            for i in tqdm.tqdm(range(sample_number * 8)):
                gsg = []
                nodes = []
                adjecent = []
                for j in range(rel_number):
                    if j == 0:
                        rel = self.sg[stage][self.random_select(
                            self.sort_sg_count[stage])[0]]
                    else:
                        rel = self.sg[stage][self.random_select(adjecent)[0]]
                    add_nodes = [rel[-1]] if len(rel) < 3 else [rel[0], rel[2]]
                    add_nodes = [k for k in add_nodes if k not in nodes]

                    for n in add_nodes:
                        adjecent += self.name2rel[stage][n]

                    nodes += add_nodes
                    gsg.append(" ".join(rel).strip())

                objects = []
                for rel in gsg[:1]:
                    rel_s = self.rel2sg[rel]
                    if len(rel_s) == 2:
                        objects.append(rel_s[-1])
                    elif len(rel_s) == 3:
                        objects += [rel_s[0], rel_s[-1]]
                    objects = list(set(objects))

                select_kg = []
                for obj in objects:
                    if obj in self.name2kg:
                        select_kg += self.name2kg[obj]
                    else:
                        for _, item in self.kb.items():
                            if obj in item[0].split() and 'a type of' not in item[1]:
                                self.name2kg.setdefault(obj, []).append(item)
                        select_kg += self.name2kg.get(obj, [])

                if select_kg:
                    generated_sg.append([" ".join(random.choice(select_kg))] + gsg)

                if len(generated_sg) == sample_number:
                    break

        elif stage in ['logical']:
            for i in tqdm.tqdm(range(int(sample_number * 1.5))):
                gsg = []
                nodes = []
                adjecent = []

                for j in range(2):
                    if j == 0:
                        att_type = random.choices(
                            ['color', 'material', 'shape'], weights=[10, 5, 1])[0]
                        rel = self.sg[stage][self.random_select(
                            self.att_type2rel[att_type])[0]]

                    else:
                        rel = self.sg[stage][self.random_select(adjecent)[0]]
                    add_nodes = [rel[0]] if len(rel) < 3 else [rel[0], rel[2]]
                    add_nodes = [k for k in add_nodes if k not in nodes]

                    for n in add_nodes:
                        adjecent += self.att2rel[n]

                    nodes += add_nodes
                    gsg.append(" ".join(rel).strip())

                if gsg[0] == gsg[1]:
                    continue

                generated_sg.append(gsg)

                if len(generated_sg) == sample_number:
                    break

        elif stage in ['textqa']:
            sg_textqa = meta['sort_sg_count']['scenetext']['OCR']
            # sample ocr relation
            for i in tqdm.tqdm(range(sample_number)):
                generated_sg.append(self.random_select(sg_textqa))

        return generated_sg

    def get_att_type2rel(self):
        att_type2rel = {}
        for att, rels in self.att2rel.items():
            flag = False
            for att_type in ['color', 'material', 'shape']:
                for c in att2type[att_type]:
                    if c in att:
                        att_type2rel.setdefault(att_type, [])
                        att_type2rel[att_type] += rels
                        flag = True
                        break
                if flag:
                    break
        return att_type2rel

    @staticmethod
    def random_select(sg_count):
        rel_names = [v for k, v in sg_count[:5000]]
        weights = [np.log(k + 1) for k, v in sg_count[:5000]]
        return random.choices(rel_names, weights)


att2type = {'color': ['white',
                      'black',
                      'red',
                      'blue',
                      'green',
                      'brown',
                      'yellow',
                      'gray',
                      'orange',
                      'pink',
                      'tan',
                      'purple',
                      'blond'],
            'material': ['wood',
                         'silver',
                         'metal',
                         'brown wood',
                         'glass',
                         'concrete',
                         'plastic',
                         'brick',
                         'gold',
                         'metal silver',
                         'paper',
                         'white ceramic',
                         'white fluffy',
                         'fluffy',
                         'snowy',
                         'stainless steel',
                         'leather'],
            'shape': ['round', 'square', 'sliced', 'long', 'short']}

if __name__ == "__main__":
    # download meta_sg_final.json and knowledge_graph.json from the link on the github
    # specify the path of meta_sg_final.json, and knowledge_graph.json
    generator = GenerateSceneGraph("./meta_sg_final.json", "./knowledge_graph.json")
    gsg = {}
    for stage in ['object', 'attribute', 'relation', 'logical', 'knowledge', 'textqa']:
        gsg[stage] = generator.generate_sg(stage, 100)

    json.dump(gsg, open("generated_sg_all_stages.json", "w"))
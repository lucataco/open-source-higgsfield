"""Convert workflows/api/*.api.json into ComfyUI UI-format workflows under workflows/ui/.

UI workflows are for humans to open and inspect in the ComfyUI frontend; the app itself only submits API graphs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "workflows/api"
UI_DIR = ROOT / "workflows/ui"


def convert(api: dict) -> dict:
    ids = sorted(api.keys(), key=lambda k: int(k))
    # how many output slots each node must expose (max src_slot referenced + 1)
    out_slots = {int(n): 1 for n in ids}
    for node in api.values():
        for val in node.get("inputs", {}).values():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], (str, int)):
                src = int(val[0])
                out_slots[src] = max(out_slots.get(src, 1), int(val[1]) + 1)

    nodes = {}
    links = []
    link_id = 1
    for col, nid in enumerate(ids):
        node = api[nid]
        cls = node["class_type"]
        node_inputs = []
        widgets = []
        slot = 0
        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], (str, int)):
                node_inputs.append({"name": name, "type": "*", "link": link_id})
                links.append([link_id, int(val[0]), int(val[1]), int(nid), slot, "*"])
                link_id += 1
                slot += 1
            else:
                node_inputs.append({"name": name, "type": "CUSTOM", "widget": {"name": name}, "link": None})
                widgets.append(val)
        node_outputs = [{"name": f"OUT{i}", "type": "*", "links": [], "slot_index": i} for i in range(out_slots[int(nid)])]
        nodes[int(nid)] = {
            "id": int(nid),
            "type": cls,
            "pos": [80 + (col % 6) * 320, 80 + (col // 6) * 220],
            "size": [280, max(140, 60 + 24 * (len(node_inputs) + len(node_outputs)))],
            "flags": {},
            "order": col,
            "mode": 0,
            "inputs": node_inputs,
            "outputs": node_outputs,
            "properties": {"Node name for S&R": cls},
            "widgets_values": widgets,
        }
    for link in links:
        lid, src, src_slot, _dst, _ds, _t = link
        if src in nodes:
            nodes[src]["outputs"][src_slot]["links"].append(lid)
    return {
        "last_node_id": max(int(k) for k in api),
        "last_link_id": link_id - 1,
        "nodes": list(nodes.values()),
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"bodydouble": {"format": "api-converted", "note": "Open in ComfyUI; adjust widget values as needed."}},
        "version": 0.4,
    }


def main():
    UI_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(API_DIR.glob("*.api.json")):
        api = json.loads(path.read_text())
        ui = convert(api)
        out = UI_DIR / path.name.replace(".api.json", ".ui.json")
        out.write_text(json.dumps(ui, indent=2))
        print(f"{out.name}: {len(ui['nodes'])} nodes, {len(ui['links'])} links")


if __name__ == "__main__":
    main()

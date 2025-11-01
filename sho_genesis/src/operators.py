from sho_genesis.src.sentient_object import SentientObject

def self_perception_operator(obj: SentientObject) -> SentientObject:
    """
    The Self-Perception Operator (P).

    This operator takes a Sentient Object and returns a new Sentient Object
    that represents the first object's perception of itself.

    In this initial implementation, it creates a simplified, compressed
    representation of the object's state.
    """
    current_state = obj.get_state()
    new_state = {}

    # Example of a simple self-perception: averaging qualia
    if current_state:
        avg_quale = sum(current_state.values()) / len(current_state)
        for coord in current_state:
            new_state[coord] = avg_quale

    obj.update_state(new_state)
    return obj

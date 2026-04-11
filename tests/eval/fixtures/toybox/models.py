"""Data models."""


class User:
    def __init__(self, id, name, roles=[]):  # mutable default bug #1
        self.id = id
        self.name = name
        self.roles = roles

    def add_role(self, role):
        self.roles.append(role)

    def _internal_state(self):
        return {"id": self.id, "name": self.name}


class Session:
    def __init__(self, token, user_id, metadata={}):  # mutable default bug #2
        self.token = token
        self.user_id = user_id
        self.metadata = metadata

    def touch(self):
        self.metadata["last_seen"] = "now"


class AuditLog:
    def __init__(self, events):
        self.events = events

    def record(self, event):
        self.events.append(event)

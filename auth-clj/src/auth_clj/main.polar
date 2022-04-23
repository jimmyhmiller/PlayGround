



allow(actor, action, resource) if
  has_permission(actor, action, resource);


has_role(user: User, name: String, resource: Resource) if
  role in user.roles and
  role.name = name and
  role.resource = resource;


actor User {}


resource Repository {
  permissions = ["view", "buy", "delete"];
  roles = ["ticketholder", "poscustomer", "admin"];
  # ticket
  "view" if "ticketholder";
  "buy" if "poscustomer";
  "delete" if "admin";
  
  "ticketholder" if "admin";
  "poscustomer" if "admin";

}


has_permission(user: User, _action: String, _repository: Repository) if
  role in user.roles and
  role.name = "admin";



# role_has_permission(role, action)  if
#   _permissions in role.permissions and
#     _permission.action = action;
# user_has_permission_via_role(role, action, resource) if
#   role.resource = resource and
#     role_has_permission(role, action);




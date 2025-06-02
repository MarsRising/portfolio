“How can we minimize the total shipment costs while ensuring capacity needs are met?”. The mathematical representations are as follows:
Objective Function:
min(∑_(i=1)^2▒∑_(j=1)^3▒〖c_ij x_ij+〗 ∑_(i=1)^2▒∑_(k=1)^65▒〖c_ik y_ik+〗 ∑_(j=1)^3▒∑_(k=1)^65▒〖c_jk z_jk 〗)

Constraint Functions:
Hub capacities:
∑_(j=1)^3▒〖x_ij+∑_(k=1)^65▒〖y_ik≤capacity,i=1,2〗〗
Quantity into focus cities:
∑_(i=1)^2▒〖x_ij≤capacity,j=1,2,3〗
Quantity out of focus cities:
∑_(k=1)^65▒〖z_jk=∑_(i=1)^2▒x_ij 〗,j=1,2,3
Center demand:
∑_(i=1)^2▒〖y_ik+∑_(j=1)^3▒〖z_jk=requirement,k=1,2,…65〗〗

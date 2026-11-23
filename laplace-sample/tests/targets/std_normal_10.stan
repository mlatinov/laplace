// Phase 1 reference target: 10-dimensional standard normal.
// Truth: E[x_i] = 0, E[x_i^2] = 1.
parameters {
  vector[10] x;
}
model {
  x ~ std_normal();
}

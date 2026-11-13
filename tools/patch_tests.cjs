const fs = require('fs');

const files = ['tests/test_agents.py', 'tests/test_experience.py'];
const pattern = /\{"model":"([a-z_]+)","parameters":(\{[^}]*\})\}/g;
const make = (model, params) =>
  `{"model":"${model}","parameters":${params},"hypothesis":"test hypothesis",` +
  `"expected_direction":"up","falsifier":"验证分数没有提升"}`;

for (const f of files) {
  let text = fs.readFileSync(f, 'utf8');
  const before = (text.match(pattern) || []).length;
  text = text.replace(pattern, (_, model, params) => make(model, params));
  fs.writeFileSync(f, text, 'utf8');
  console.log(f, 'patched', before, 'occurrences');
}

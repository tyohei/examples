---
Date: 2019-09-29
Name: Yohei Tsuji
---
# TypeScript

```sh
node --version  # v12.10.0
npm --version  # 6.11.3
npm install typescript
```

```sh
npx tsc --version  # Version 3.6.3
```

```sh
vim greeter.ts
```

```typescript
function greeter(person) {
    return "Hello, " + person;
}

let user = "Jane User";

document.body.textContent = greeter(user);
```

```sh
npx tsc greeter.ts
```

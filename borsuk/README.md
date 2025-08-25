# Borsuk – mini język + interpreter w Pythonie

**Pliki:**
- `borsuk.py` — interpreter + REPL + przykłady
- `borsuk_language_doc.pdf` — skrócony podręcznik PDF

## Szybki start
```bash
python3 borsuk.py
```
W REPL wpisuj kod, zatwierdź pustą linią, aby wykonać.

## Uruchamianie pliku
```bash
python3 borsuk.py program.bk
```

## Przykład
```
let x = 5; x += 2; x *= 3; print("x=", x)
class Osoba {
  func init(imie, wiek) { this.imie = imie; this.wiek = wiek; }
  func hello() { print("Cześć, jestem ${this.imie} (${this.wiek})"); }
}
let p = new Osoba("Ala", 20); p.hello();
```

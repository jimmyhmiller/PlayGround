FROM clojure:tools-deps-1.9.0.394-alpine

EXPOSE 8080
COPY deps.edn .

RUN clj -e ""

COPY . .

ENTRYPOINT clj -A:micro  

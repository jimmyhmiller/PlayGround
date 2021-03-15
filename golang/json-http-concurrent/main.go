package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"sync"
)

type Server struct {
	jobsQueue      chan post
	numWorkers     int
	commentsByPost *sync.Map
}

type post struct {
	UserId int    `json:"userId"`
	Id     int    `json:"id"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

type comment struct {
	PostID int    `json:"postId"`
	ID     int    `json:"id"`
	Name   string `json:"name"`
	Email  string `json:"email"`
	Body   string `json:"body"`
}

func doWork(jobsQueue <-chan post, commentsByPost *sync.Map) {
	for {
		post := <-jobsQueue
		resp, err := http.Get(fmt.Sprintf("https://jsonplaceholder.typicode.com/posts/%d/comments", post.Id))
		if err != nil {
			log.Fatalln(err)
		}
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			log.Fatal(err)
		}

		var comments []comment
		json.Unmarshal([]byte(body), &comments)
		commentsByPost.Store(post.Id, comments)
	}
}

func (server *Server) submitWork(p post) {
	server.jobsQueue <- p
}

func (server *Server) startWorkers() {
	for i := 0; i < server.numWorkers; i++ {
		go doWork(server.jobsQueue, server.commentsByPost)
	}
}

func (server *Server) fetchPosts(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get("https://jsonplaceholder.typicode.com/posts")
	if err != nil {
		log.Fatalln(err)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}

	var posts []post
	json.Unmarshal([]byte(body), &posts)
	go func() {
		for i, p := range posts {
			log.Println(i)
			server.submitWork(p)
		}
	}()
	w.Header().Add("Content-Type", "application/json")
	val, err := json.Marshal(struct {
		Message string `json:"message"`
	}{
		Message: fmt.Sprintf("Found %d posts", len(posts)),
	})
	w.Write(val)
}

func (server *Server) getComments(w http.ResponseWriter, r *http.Request) {
	key, err := strconv.Atoi(r.URL.Query().Get("id"))
	if err != nil {
		log.Fatal(err)
	}

	val, ok := server.commentsByPost.Load(key)
	if ok == false {
		http.NotFound(w, r)
	} else {
		b, err := json.Marshal(val)
		if err != nil {
			http.Error(w, "invalid comments", 500)
		}
		w.Header().Add("Content-Type", "application/json")
		w.Write(b)
	}
}

func main() {
	var commentsByPost sync.Map
	s := Server{
		jobsQueue:      make(chan post, 10),
		numWorkers:     2,
		commentsByPost: &commentsByPost,
	}
	s.startWorkers()

	http.HandleFunc("/fetchPosts", s.fetchPosts)
	http.HandleFunc("/commentsByPost", s.getComments)
	log.Println("Starting Server")

	log.Fatal(http.ListenAndServe(":8080", nil))
}

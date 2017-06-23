const chalk = require('chalk')

// ${chalk.green('http')} POST https://insta-rest.io/api name="my-api"

const tutorial = `\
Thanks for signing up for insta-rest. You now have a fully working rest api. Your api has lots of features, too many to show just now, but here are some of the basics.

    ${chalk.blue('Get collection of entities')}
    ${chalk.green('GET')} https://insta-rest.io/api/my-api/:entity 

    ${chalk.blue('Get single entity')}
    ${chalk.green('GET')} https://insta-rest.io/api/my-api/:entity/:id 

    ${chalk.blue('Update entity (full)')}
    ${chalk.green('PUT')} https://insta-rest.io/api/my-api/:entity/:id 

    ${chalk.blue('Update entity (partial)')}
    ${chalk.green('PATCH')} https://insta-rest.io/api/my-api/:entity/:id 

    ${chalk.blue('Create new entity')}
    ${chalk.green('POST')} https://insta-rest.io/api/my-api/:entity 

    ${chalk.blue('Delete an entity')}
    ${chalk.green('DELETE')} https://insta-rest.io/api/my-api/:entity/:id

With those endpoints you have a fully working api. Here are more features that built-in to your api.

  ${chalk.green('*')} Sort, Query, and Paginate
  ${chalk.green('*')} Authentication
  ${chalk.green('*')} Validations
  ${chalk.green('*')} Automatically Suggested Validations
  ${chalk.green('*')} Generate Example Data
  ${chalk.green('*')} Generate Swagger Documentation
  ${chalk.green('*')} Custom Triggered Functions

To learn about all these features you can:

1) Read our documentation [https://insta-rest.io/docs]
2) Follow the interactive tutorial. 

   ${chalk.green('http')} https://insta-rest.io/api/my-api/_tutorial

${chalk.white.bold('Next Steps:')} Consider adding an email address. You are on the free tier right now, by adding your email you will be able to be alerted when you are about to hit your limit and be able to upgrade to a paid account. We will not spam you.

    ${chalk.green('http')} post https://insta-rest.io/api/my-api/_account emailAddress="example@example.com"
`

console.log('\x1Bc')
console.log(tutorial)

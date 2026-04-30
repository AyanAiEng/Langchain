from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    separator="\n\n",   
    chunk_size=1000,
    chunk_overlap=200
)

text = """Lorem ipsum dolor sit amet consectetur adipisicing elit. Ducimus perspiciatis as
periores consequuntur voluptatem officia officiis magnam eum, vitae delectus del





zeniti nostr
um, porro amet deserunt beatae ad? Sit omnis adipisci commodi incidunt officiis nesciunt accusam
us ipsum reiciendis illo laborum quae neque dolores quam consequuntur ducimus ab fuga unde quibusdam



non expedita, consectetur rerum ipsam rem eligendi! Debitis eaque architecto i

ncidunt impedit necessitatibus! Dignissimos nisi nesciunt cupiditate minima vero aut, odio at fugiat quis neque, hic expedita doloribus veniam quaerat eum illo fuga dolor ut voluptatibus, voluptatem quas maiores accusantium saepe reprehenderit! Quo, animi ducimus rem repudiandae libero aspernatur maxime totam non ipsam dolores commodi est minus eaque deleniti consequuntur laboriosam, tenetur asperiores dignissimos ut. Libero, sequi impedit aliquam molestiae iure dolores iusto odit corporis ipsa


aspernatur quidem saepe nemo, itaque nobis, soluta mollitia autem ipsum! Nesciunt, dolores repudiandae ipsam facilis cum explicabo ad provident dignissimos mollitia aliquam incidunt dolorum suscipit optio itaque quod ea sunt, ab voluptas perferendis. Blanditiis sit in similique dolorem, qui dolore alias. Dolores consequatur amet impedit odio sapiente necessitatibus perferendis dicta et recusandae ex odit autem sint facilis magni quidem est reprehenderit ut doloremque enim ipsam non, minima suscipit. Non quidem consequatur unde illum accusantium quaerat laboriosam distinctio assumenda


nesciunt commodi voluptas perspiciatis iure sunt et, vero eaque officiis soluta! Officiis, dolorem! Corrupti repudiandae, debitis magnam adipisci eos blanditiis, impedit rem velit facilis nemo culpa vero tempore. Dolorum repellat accusantium autem adipisci reiciendis at officia facere vel ratione nulla? Facilis nobis possimus accusantium maxime repellat suscipit odit praesentium accusamus, pariatur sapiente! Esse dolor cum magnam id est. Exercitationem, magni hic. Dicta in ipsam quas, magni ad architecto alias iste id quidem, dignissimos, aspernatur animi sunt ratione duci


mus? Illo temporibus obcaecati voluptate, corporis, dolores a sed autem iure neque eveniet, quis inventore. Reiciendis dicta et necessitatibus quos quaerat!"""

chunks = splitter.split_text(text)

print(chunks)
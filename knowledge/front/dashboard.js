const titles={query:"智能问答",documents:"文档管理",storage:"存储数据",evaluation:"评测中心"};
document.querySelectorAll(".nav-item").forEach(button=>{
  button.addEventListener("click",()=>{
    document.querySelectorAll(".nav-item").forEach(item=>item.classList.remove("active"));
    document.querySelectorAll(".view").forEach(view=>view.classList.remove("active"));
    button.classList.add("active");
    const name=button.dataset.view;
    document.querySelector(`#view-${name}`).classList.add("active");
    document.querySelector("#pageTitle").textContent=titles[name];
  });
});
